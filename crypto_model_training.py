import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# 1. 시계열 윈도우 커스텀 데이터셋
class CryptoTimeSeriesDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# 2. 멀티 브랜치 신경망
class MultiBranchCryptoPredictor(nn.Module):
    def __init__(self, num_indicators, hidden_price=64, hidden_vol=32, hidden_ind=64, dropout=0.3):
        super(MultiBranchCryptoPredictor, self).__init__()
        
        self.lstm_price = nn.LSTM(4, hidden_price, num_layers=2, batch_first=True, dropout=dropout)
        self.lstm_vol = nn.LSTM(1, hidden_vol, num_layers=2, batch_first=True, dropout=dropout)
        self.lstm_ind = nn.LSTM(num_indicators, hidden_ind, num_layers=2, batch_first=True, dropout=dropout)

        self.bn_price = nn.BatchNorm1d(hidden_price)
        self.bn_vol = nn.BatchNorm1d(hidden_vol)
        self.bn_ind = nn.BatchNorm1d(hidden_ind)

        combined_size = hidden_price + hidden_vol + hidden_ind
        self.meta_learner = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3) 
        )

    def forward(self, x):
        x_price = x[:, :, 0:4]   
        x_vol   = x[:, :, 4:5]   
        x_ind   = x[:, :, 5:]    

        out_price, _ = self.lstm_price(x_price)
        feat_price = self.bn_price(out_price[:, -1, :])

        out_vol, _ = self.lstm_vol(x_vol)
        feat_vol = self.bn_vol(out_vol[:, -1, :])

        out_ind, _ = self.lstm_ind(x_ind)
        feat_ind = self.bn_ind(out_ind[:, -1, :])

        combined_features = torch.cat((feat_price, feat_vol, feat_ind), dim=1)
        return self.meta_learner(combined_features)


# 3. 데이터 파이프라인: 동적 라벨링 적용
def prepare_data(filepath, seq_length=120):
    print(f"[INFO] 학습 데이터 로드 중: {filepath}")
    df = pd.read_csv(filepath)

    if 'atr' in df.columns:
        df.drop(columns=['atr'], inplace=True)

    raw_filepath = filepath.replace("_processed.csv", "_5m_raw.csv")
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(f"[ERROR] 원본 데이터가 필요합니다: {raw_filepath}")
    
    df_raw = pd.read_csv(raw_filepath)
    df = pd.merge(df, df_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']], on='timestamp', suffixes=('', '_raw'))

    # 1시간봉 추세 피처
    df['1h_ema_50'] = df['close_raw'].ewm(span=12 * 50, adjust=False).mean()
    df['1h_ema_200'] = df['close_raw'].ewm(span=12 * 200, adjust=False).mean()
    df['1h_trend'] = np.where(df['1h_ema_50'] > df['1h_ema_200'], 1, -1)

    # 시간 주기 피처
    dt = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)

    # 🌟 [GPT 조언 반영] 정답지 생성을 위한 % 단위 ATR (변동성) 직접 계산
    prev_close = df['close_raw'].shift(1)
    tr1 = df['high_raw'] - df['low_raw']
    tr2 = (df['high_raw'] - prev_close).abs()
    tr3 = (df['low_raw'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['atr_pct'] = (atr / df['close_raw']) * 100
    df['atr_pct'].fillna(method='bfill', inplace=True)

    # 🌟 [동적 트리플 배리어 라벨링] 장세에 따라 익절/손절선이 춤을 춥니다!
    horizon = 72
    close_prices = df['close_raw'].values
    atr_pct_vals = df['atr_pct'].values
    n = len(close_prices)
    targets = np.zeros(n, dtype=int) 

    for i in range(n - horizon):
        curr_p = close_prices[i]
        curr_atr = atr_pct_vals[i]
        
        # 동적 배리어 폭 산정 (익절: ATR의 2.0배, 손절: ATR의 1.0배 적용하되 최소치 보장)
        tp_thresh = max(curr_atr * 2.0, 0.6)  # 최소 0.6%
        sl_thresh = max(curr_atr * 1.0, 0.3)  # 최소 0.3%

        future_window = close_prices[i+1 : i+1+horizon]
        ret = (future_window - curr_p) / curr_p * 100
        
        hit_tp_long = np.where(ret >= tp_thresh)[0]
        hit_sl_long = np.where(ret <= -sl_thresh)[0]
        idx_tp_long = hit_tp_long[0] if len(hit_tp_long) > 0 else horizon + 1
        idx_sl_long = hit_sl_long[0] if len(hit_sl_long) > 0 else horizon + 1
        is_long = (idx_tp_long < idx_sl_long) and (idx_tp_long <= horizon)
        
        hit_tp_short = np.where(ret <= -tp_thresh)[0]
        hit_sl_short = np.where(ret >= sl_thresh)[0]
        idx_tp_short = hit_tp_short[0] if len(hit_tp_short) > 0 else horizon + 1
        idx_sl_short = hit_sl_short[0] if len(hit_sl_short) > 0 else horizon + 1
        is_short = (idx_tp_short < idx_sl_short) and (idx_tp_short <= horizon)

        if is_long and not is_short:
            targets[i] = 1 
        elif is_short and not is_long:
            targets[i] = 2 
        elif is_long and is_short:
            targets[i] = 1 if idx_tp_long < idx_tp_short else 2

    df['Target'] = targets

    # 가격/거래량 스케일링
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = df[f'{col}_raw'].pct_change().fillna(0)
        q_lo, q_hi = df[col].quantile(0.001), df[col].quantile(0.999)
        df[col] = df[col].clip(q_lo, q_hi)
    
    vol_ma = df['volume_raw'].rolling(24).mean() + 1e-9
    vol_col = ['volume']
    df[vol_col[0]] = (df['volume_raw'] / vol_ma).clip(0, 10) 

    # 🌟 [요청사항 완벽 반영] 라벨링에만 쓰고, 학습 피처에서는 완벽히 제거
    drop_cols = [c for c in df.columns if c.endswith('_raw')] + ['atr_pct']
    df.drop(columns=drop_cols, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    exclude_cols = ['timestamp', 'datetime', 'Target', '1h_ema_50', '1h_ema_200']
    ind_cols = [col for col in df.columns if col not in price_cols + vol_col + exclude_cols]
    
    feature_cols = price_cols + vol_col + ind_cols
    features = df[feature_cols].values
    targets = df['Target'].values

    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)

    X_train, y_train = features[:train_end], targets[:train_end]
    X_val, y_val = features[train_end:val_end], targets[train_end:val_end]
    X_test, y_test = features[val_end:], targets[val_end:]

    class_counts = np.bincount(y_train, minlength=3)
    class_weights = len(y_train) / (3.0 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print(f"[INFO] 🎯 동적 배리어 적용 완료 (최소 익절 0.6% / 손절 0.3%)")
    print(f"[INFO] 전문가 분할 - 가격(4), 거래량(1), 보조지표({len(ind_cols)})")
    print(f"[INFO] 라벨 분포 - 관망(0): {class_counts[0]} | 롱(1): {class_counts[1]} | 숏(2): {class_counts[2]}")

    train_dataset = CryptoTimeSeriesDataset(X_train, y_train, seq_length)
    val_dataset = CryptoTimeSeriesDataset(X_val, y_val, seq_length)
    test_dataset = CryptoTimeSeriesDataset(X_test, y_test, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train_loader, val_loader, test_loader, len(ind_cols), class_weights

# 4. 실전 지표 기반 모델 트레이너 (F1-Score Monitoring)
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device, class_weights, patience=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 🌟 Loss가 아닌 F1 Score가 높아지는 쪽으로 LR을 조절 (mode='max')
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
        self.patience = patience

    def train(self, epochs=100):
        print(f"[INFO] 하드웨어 가속기({self.device}) 기반 다중 브랜치 학습 시작...")
        best_val_f1 = -1.0
        best_model_weights = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)

            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_trues = []
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    predictions = self.model(X_batch)
                    loss = self.criterion(predictions, y_batch)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(predictions.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_trues.extend(y_batch.cpu().numpy())

            avg_val_loss = val_loss / len(self.val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 🌟 실전 지표: 롱(1)과 숏(2)의 Macro F1 Score 계산
            val_f1 = f1_score(val_trues, val_preds, labels=[1, 2], average='macro', zero_division=0)
            
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val F1(Long/Short): {val_f1:.4f} | LR: {current_lr:.6f}")

            self.scheduler.step(val_f1)

            # 조기 종료 기준도 F1 Score로 변경!
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_weights = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n[INFO] {self.patience} Epoch 동안 롱/숏 F1-Score 개선이 없어 조기 종료합니다!")
                    break

        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
        return self.model

# 5. 최종 모델 평가
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    print(f"\n[RESULT] ========================================")
    print("분류 리포트 (0: 관망, 1: 롱 진입, 2: 숏 진입)")
    print(classification_report(all_labels, all_preds, labels=[0, 1, 2], target_names=['Hold(0)', 'Long(1)', 'Short(2)'], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]))
    print(f"=================================================\n")

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    filepath = "data/BTC_USDT_processed.csv"

    if not os.path.exists(filepath):
        print(f"[ERROR] '{filepath}' 파일을 찾을 수 없습니다.")
    else:
        train_loader, val_loader, test_loader, num_indicators, class_weights = prepare_data(filepath, seq_length=120)
        
        model = MultiBranchCryptoPredictor(num_indicators=num_indicators, dropout=0.3)
        trainer = ModelTrainer(model, train_loader, val_loader, device, class_weights=class_weights, patience=10)
        trained_model = trainer.train(epochs=100)
        evaluate_model(trained_model, test_loader, device)

        os.makedirs("models", exist_ok=True)
        save_path = "models/best_lstm_btc_5m_multibranch_v4.pth"
        torch.save(trained_model.state_dict(), save_path)
        print(f"[INFO] 💾 다중 브랜치 앙상블 모델 저장 완료: {save_path}")