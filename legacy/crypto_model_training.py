import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import talib
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

# 2. 4개 브랜치 기반 신경망
class MultiBranchCryptoPredictor(nn.Module):
    def __init__(self, num_indicators, num_patterns, hidden_price=64, hidden_vol=32, hidden_ind=64, hidden_pat=32, dropout=0.3):
        super(MultiBranchCryptoPredictor, self).__init__()
        
        self.num_ind = num_indicators
        self.num_pat = num_patterns

        # 4명의 전문가 (Branches)
        self.lstm_price = nn.LSTM(4, hidden_price, num_layers=2, batch_first=True, dropout=dropout)
        self.lstm_vol = nn.LSTM(1, hidden_vol, num_layers=2, batch_first=True, dropout=dropout)
        self.lstm_ind = nn.LSTM(num_indicators, hidden_ind, num_layers=2, batch_first=True, dropout=dropout)
        self.lstm_pat = nn.LSTM(num_patterns, hidden_pat, num_layers=2, batch_first=True, dropout=dropout)  # 패턴 전문가 브랜치

        # 정규화 레이어
        self.bn_price = nn.BatchNorm1d(hidden_price)
        self.bn_vol = nn.BatchNorm1d(hidden_vol)
        self.bn_ind = nn.BatchNorm1d(hidden_ind)
        self.bn_pat = nn.BatchNorm1d(hidden_pat)

        # 메타 러너 (통합 결정권자)
        combined_size = hidden_price + hidden_vol + hidden_ind + hidden_pat
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
        # 입력 데이터를 4개의 특성으로 정확히 분할
        x_price = x[:, :, 0:4]   
        x_vol   = x[:, :, 4:5]   
        x_ind   = x[:, :, 5 : 5 + self.num_ind]    
        x_pat   = x[:, :, 5 + self.num_ind :]  # 패턴 데이터

        out_price, _ = self.lstm_price(x_price)
        feat_price = self.bn_price(out_price[:, -1, :])

        out_vol, _ = self.lstm_vol(x_vol)
        feat_vol = self.bn_vol(out_vol[:, -1, :])

        out_ind, _ = self.lstm_ind(x_ind)
        feat_ind = self.bn_ind(out_ind[:, -1, :])

        out_pat, _ = self.lstm_pat(x_pat)
        feat_pat = self.bn_pat(out_pat[:, -1, :])

        # 4개 전문가의 의견 병합
        combined_features = torch.cat((feat_price, feat_vol, feat_ind, feat_pat), dim=1)
        return self.meta_learner(combined_features)


# 3. 데이터 파이프라인: 캔들 패턴 추출 로직 추가
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

    # TA-Lib 10대 핵심 캔들 패턴 추출 (원본 가격 기준)
    o = df['open_raw'].values
    h = df['high_raw'].values
    l = df['low_raw'].values
    c = df['close_raw'].values

    # TA-Lib는 패턴 발견 시 100 또는 -100을 반환하므로, 신경망에 맞게 1.0 / -1.0으로 스케일링
    df['pat_doji']         = talib.CDLDOJI(o, h, l, c) / 100.0
    df['pat_hammer']       = talib.CDLHAMMER(o, h, l, c) / 100.0
    df['pat_engulfing']    = talib.CDLENGULFING(o, h, l, c) / 100.0
    df['pat_morningstar']  = talib.CDLMORNINGSTAR(o, h, l, c) / 100.0
    df['pat_eveningstar']  = talib.CDLEVENINGSTAR(o, h, l, c) / 100.0
    df['pat_shootingstar'] = talib.CDLSHOOTINGSTAR(o, h, l, c) / 100.0
    df['pat_hangingman']   = talib.CDLHANGINGMAN(o, h, l, c) / 100.0
    df['pat_piercing']     = talib.CDLPIERCING(o, h, l, c) / 100.0
    df['pat_darkcloud']    = talib.CDLDARKCLOUDCOVER(o, h, l, c) / 100.0
    df['pat_harami']       = talib.CDLHARAMI(o, h, l, c) / 100.0

    # 고정 트리플 배리어 라벨링 (백테스트와 동일: TP=1.4%, SL=0.7%, 손익비 2:1)
    horizon = 72
    tp_thresh = 1.4   # 1.4% 고정 (백테스트 tp_pct=0.014)
    sl_thresh = 0.7   # 0.7% 고정 (백테스트 sl_pct=0.007)
    close_prices = df['close_raw'].values
    n = len(close_prices)
    targets = np.zeros(n, dtype=int) 

    for i in range(n - horizon):
        curr_p = close_prices[i]

        future_window = close_prices[i+1: i+1+horizon]
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

    # Hold 다운샘플링 — Long/Short 평균 개수에 맞춰 Hold 축소 (시퀀스 순서 보존)
    hold_idx  = np.where(targets == 0)[0]
    long_idx  = np.where(targets == 1)[0]
    short_idx = np.where(targets == 2)[0]
    target_hold_n = (len(long_idx) + len(short_idx)) // 2
    if len(hold_idx) > target_hold_n:
        rng = np.random.default_rng(42)
        hold_idx = rng.choice(hold_idx, size=target_hold_n, replace=False)
        keep_idx = np.sort(np.concatenate([hold_idx, long_idx, short_idx]))
        df = df.iloc[keep_idx].reset_index(drop=True)
        targets = targets[keep_idx]
    new_counts = np.bincount(targets, minlength=3)
    print(f"[INFO] Hold 다운샘플링 후 라벨 분포 - 관망(0): {new_counts[0]} | 롱(1): {new_counts[1]} | 숏(2): {new_counts[2]}")

    # 가격/거래량 스케일링
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = df[f'{col}_raw'].pct_change().fillna(0)
        q_lo, q_hi = df[col].quantile(0.001), df[col].quantile(0.999)
        df[col] = df[col].clip(q_lo, q_hi)
    
    vol_ma = df['volume_raw'].rolling(24).mean() + 1e-9
    vol_col = ['volume']
    df[vol_col[0]] = (df['volume_raw'] / vol_ma).clip(0, 10) 

    # 불필요 원본 컬럼 제거
    drop_cols = [c for c in df.columns if c.endswith('_raw')]
    df.drop(columns=drop_cols, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 피처 순서 고정 (가격 -> 거래량 -> 보조지표 -> 패턴)
    exclude_cols = ['timestamp', 'datetime', 'Target', '1h_ema_50', '1h_ema_200']
    pat_cols = [c for c in df.columns if c.startswith('pat_')]
    ind_cols = [c for c in df.columns if c not in price_cols + vol_col + pat_cols + exclude_cols]
    
    feature_cols = price_cols + vol_col + ind_cols + pat_cols
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

    print(f"[INFO] 🎯 V5 캔들 패턴 10종 장착 완료")
    print(f"[INFO] 전문가 분할 - 가격(4), 거래량(1), 보조지표({len(ind_cols)}), 캔들패턴({len(pat_cols)})")
    print(f"[INFO] 라벨 분포 - 관망(0): {class_counts[0]} | 롱(1): {class_counts[1]} | 숏(2): {class_counts[2]}")

    train_dataset = CryptoTimeSeriesDataset(X_train, y_train, seq_length)
    val_dataset = CryptoTimeSeriesDataset(X_val, y_val, seq_length)
    test_dataset = CryptoTimeSeriesDataset(X_test, y_test, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train_loader, val_loader, test_loader, len(ind_cols), len(pat_cols), class_weights

# 4. Focal Loss: 어려운 샘플에 더 집중하는 손실 함수
class FocalLoss(nn.Module):
    """
    Focal Loss = -α * (1 - p_t)^γ * log(p_t)
    
    Parameters:
    - gamma: focusing parameter (기본 2.0)
    - class_weights: 클래스 가중치 (불균형 해결)
    - label_smoothing: 라벨 스무딩
    """
    def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # 기본 크로스엔트로피 손실 (라벨 스무딩 포함)
        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        # 확률 계산: p_t = exp(-ce_loss)
        p = torch.exp(-ce_loss)
        
        # Focal Loss = (1-p)^γ * ce_loss
        # γ=0: 기본 CE와 동일
        # γ>0: 쉬운 샘플(p 높음)은 다운웨이팅, 어려운 샘플(p 낮음)은 업웨이팅
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        return focal_loss.mean()

# 5. 실전 지표 기반 모델 트레이너
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device, class_weights, patience=10, use_focal_loss=True, gamma=2.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_focal_loss = use_focal_loss
        
        # Focal Loss 또는 CrossEntropyLoss 선택
        if use_focal_loss:
            self.criterion = FocalLoss(gamma=gamma, class_weights=class_weights.to(device), label_smoothing=0.1)
            print(f"[INFO] 🎯 Focal Loss 적용 (γ={gamma}, 어려운 샘플에 가중 집중)")
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
            print(f"[INFO] CrossEntropyLoss 적용")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
        self.patience = patience

    def train(self, epochs=100):
        print(f"[INFO] 하드웨어 가속기({self.device}) 기반 4-Branch 학습 시작...")
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
            
            val_f1 = f1_score(val_trues, val_preds, labels=[1, 2], average='macro', zero_division=0)
            
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val F1(Long/Short): {val_f1:.4f} | LR: {current_lr:.6f}")

            self.scheduler.step(val_f1)

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

# 6. 최종 모델 평가
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
    # 재현성을 위한 시드 고정
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    import random; random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    filepath = "data/BTC_USDT_processed.csv"

    if not os.path.exists(filepath):
        print(f"[ERROR] '{filepath}' 파일을 찾을 수 없습니다.")
    else:
        # V5 로드
        train_loader, val_loader, test_loader, num_ind, num_pat, class_weights = prepare_data(filepath, seq_length=120)
        
        # 모델 파라미터 업데이트
        model = MultiBranchCryptoPredictor(num_indicators=num_ind, num_patterns=num_pat, dropout=0.3)
        trainer = ModelTrainer(model, train_loader, val_loader, device, class_weights=class_weights, patience=10)
        trained_model = trainer.train(epochs=100)
        evaluate_model(trained_model, test_loader, device)

        os.makedirs("models", exist_ok=True)
        save_path = "models/best_lstm_btc_5m_v5.pth"
        torch.save(trained_model.state_dict(), save_path)
        print(f"[INFO] 💾 V5 모델 저장 완료: {save_path}")