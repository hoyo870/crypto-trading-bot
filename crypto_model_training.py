import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. 시계열 윈도우 생성을 위한 커스텀 데이터셋 클래스
class CryptoTimeSeriesDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        # seq_length(예: 120)만큼의 과거 데이터를 윈도우로 묶음
        x = self.features[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 2. Focal Loss: 쉬운 샘플의 기여도를 줄이고 어려운 샘플에 집중하는 손실 함수
class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    gamma > 0 이면 잘 분류된 샘플의 가중치를 낮춰 어려운 샘플에 집중한다.
    alpha는 클래스 불균형 보정 가중치 (pos_weight 역할).
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)  # 정답 클래스의 확률
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# 3. Attention 레이어
class Attention(nn.Module):
    """LSTM 출력 전체에 대해 중요도 가중치를 학습하는 Attention 메커니즘."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        scores = self.attn(lstm_out)          # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1) # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)  # (batch, hidden)
        return context

# 4. LSTM + Attention 딥러닝 모델 아키텍처
class CryptoPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(CryptoPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Attention: 모든 시점의 hidden state를 가중 합산
        context = self.attention(lstm_out)      # (batch, hidden)
        context = self.bn(context)
        context = self.dropout(context)
        out = self.fc(context)
        return out.squeeze()  # (batch_size,)

# 5. 데이터 로딩 및 정답지(Label) 생성 로직 (리스크 관리 Triple Barrier 적용)
def prepare_data(filepath, seq_length=120):
    print(f"[INFO] 학습 데이터 로드 중: {filepath}")
    df = pd.read_csv(filepath)

    # 시간 주기 피처: 시간대·요일을 sin/cos로 인코딩하여 순환 패턴 학습
    if 'datetime' in df.columns:
        dt = pd.to_datetime(df['datetime'])
    else:
        dt = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    hour = dt.dt.hour
    dow  = dt.dt.dayofweek  # 0=월 ~ 6=일
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin']  = np.sin(2 * np.pi * dow  / 7)
    df['dow_cos']  = np.cos(2 * np.pi * dow  / 7)

    # 🌟 [리스크 관리(Triple Barrier) 라벨링 적용] 🌟
    tp_pct = 2.0   # 익절선: +2%
    sl_pct = -1.0  # 손절선: -1%
    horizon = 72   # 최대 대기 시간: 72봉 (5분봉 기준 6시간)
    
    print(f"[INFO] 🎯 리스크 관리 라벨링 중: 익절 +{tp_pct}%, 손절 {sl_pct}%, 최대 대기 {horizon}봉")
    
    close_prices = df['close'].values
    n = len(close_prices)
    targets = np.full(n, np.nan) 

    # Numpy를 활용한 고속 연산 (미래 N봉 탐색)
    for i in range(n - horizon):
        curr_p = close_prices[i]
        future_window = close_prices[i+1 : i+1+horizon]
        
        # 미래 N개의 수익률 계산
        future_ret = (future_window - curr_p) / curr_p * 100
        
        # 익절/손절선 도달 시점(인덱스) 찾기
        hit_tp = np.where(future_ret >= tp_pct)[0]
        hit_sl = np.where(future_ret <= sl_pct)[0]
        
        idx_tp = hit_tp[0] if len(hit_tp) > 0 else horizon + 1
        idx_sl = hit_sl[0] if len(hit_sl) > 0 else horizon + 1
        
        if idx_tp < idx_sl and idx_tp <= horizon:
            targets[i] = 1  # 손절 안 당하고 익절선 먼저 터치 -> 매수(1)
        else:
            targets[i] = 0  # 손절선 터치 혹은 시간 내 도달 못 함 -> 매수 안 함(0)

    df['Target'] = targets
    df.dropna(subset=['Target'], inplace=True)
    df['Target'] = df['Target'].astype(int)
    
    # 🌟 [수익률 변환] 가격 및 볼륨 데이터를 변화율(%)로 변환
    price_volume_cols = ['open', 'high', 'low', 'close', 'volume']
    df[price_volume_cols] = df[price_volume_cols].pct_change().fillna(0)

    # 이상치 클리핑: 0.1% ~ 99.9% 분위수 범위로 제한하여 극단값의 영향 제거
    for col in price_volume_cols:
        q_lo = df[col].quantile(0.001)
        q_hi = df[col].quantile(0.999)
        df[col] = df[col].clip(q_lo, q_hi)

    # 미래 데이터를 당겨오면서 발생한 결측치 행 제거
    df.dropna(inplace=True)

    # 모델 학습에 불필요한 날짜/타겟 컬럼 제외하고 순수 피처만 추출
    exclude_cols = ['timestamp', 'datetime', 'Target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    features = df[feature_cols].values
    targets = df['Target'].values

    # [데이터셋 분리] Time-Series Split (미래 데이터 섞임 방지를 위해 순서 유지)
    n = len(df)
    train_end = int(n * 0.70)  # 70% Train
    val_end = int(n * 0.85)    # 15% Validation

    X_train, y_train = features[:train_end], targets[:train_end]
    X_val, y_val = features[train_end:val_end], targets[train_end:val_end]
    X_test, y_test = features[val_end:], targets[val_end:]

    # 클래스 불균형 처리 (1의 개수가 적을 테니 가중치를 부여)
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32)
    print(f"[INFO] 클래스 분포 - 진입 찬스(1): {pos_count} | 관망/위험(0): {neg_count} | pos_weight: {pos_weight.item():.4f}")

    # DataLoader 생성
    train_dataset = CryptoTimeSeriesDataset(X_train, y_train, seq_length)
    val_dataset = CryptoTimeSeriesDataset(X_val, y_val, seq_length)
    test_dataset = CryptoTimeSeriesDataset(X_test, y_test, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"[INFO] 모델 입력 피처 개수: {len(feature_cols)}개 (수익률 변환 완료)")
    print(f"[INFO] 분할 완료 - Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, len(feature_cols), pos_weight

# 6. 학습 루프 및 조기 종료 (Early Stopping) 관리 클래스
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device, patience=10, pos_weight=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Focal Loss: 클래스 불균형 + 어려운 샘플 집중
        alpha = pos_weight.item() if pos_weight is not None else 1.0
        self.criterion = FocalLoss(alpha=alpha, gamma=2.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

        # 검증 손실이 개선되지 않으면 LR을 절반으로 감소
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        self.patience = patience

    def train(self, epochs=100):
        print(f"[INFO] 하드웨어 가속기({self.device}) 기반 학습 시작...")
        best_val_loss = float('inf')
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

                # Gradient Clipping: 기울기 폭발 방지
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)

            # 검증 (Validation) 루프
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    predictions = self.model(X_batch)
                    loss = self.criterion(predictions, y_batch)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

            # LR Scheduler: 검증 손실 기반으로 학습률 조정
            self.scheduler.step(avg_val_loss)

            # 조기 종료 (Early Stopping) 체크
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n[INFO] {self.patience} Epoch 동안 검증 손실이 개선되지 않아 조기 종료(Early Stop)합니다.")
                    break

        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
        return self.model


# 7. 최적 분류 임계값 탐색 (검증셋 기반)
def find_optimal_threshold(model, val_loader, device):
    """
    검증셋에서 F1-Score가 최대인 임계값을 탐색한다.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            probs = torch.sigmoid(model(X_batch)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.3, 0.95, 0.01):
        preds = (all_probs >= thresh).astype(int)
        score = f1_score(all_labels, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, thresh

    print(f"[INFO] 최적 임계값 탐색 완료: {best_thresh:.2f} (Val F1: {best_f1:.4f})")
    # 임계값을 보수적으로 약간 높게 조정 (가짜 신호 방어)
    return min(best_thresh + 0.05, 0.90)


# 8. 최종 모델 평가 함수
def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities >= threshold).float()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy  = 100 * (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds)

    print(f"\n[RESULT] ========================================")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  Precision(적중률) : {precision:.4f} -> 내가 1이라 찍었을 때 실제로 맞출 확률")
    print(f"  Recall(탐지율)    : {recall:.4f} -> 전체 기회 중 놓치지 않고 잡아낸 비율")
    print(f"  F1-Score         : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"[RESULT] ========================================\n")

if __name__ == "__main__":
    # Mac M1/M2/M3 하드웨어 가속기(MPS) 연결
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1단계 파이프라인에서 추출된 '전처리 완료' 파일명 지정
    filepath = "data/BTC_USDT_processed.csv"

    # 파일이 존재하는지 검증
    if not os.path.exists(filepath):
        print(f"[ERROR] '{filepath}' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
    else:
        # 과거 120개 캔들(10시간)을 보고 매매 적합성 평가
        train_loader, val_loader, test_loader, input_size, pos_weight = prepare_data(filepath, seq_length=120)

        # 모델 세팅
        model = CryptoPredictorLSTM(input_size=input_size, hidden_size=256, num_layers=3, dropout=0.3)

        # 학습 시작
        trainer = ModelTrainer(model, train_loader, val_loader, device, patience=10, pos_weight=pos_weight)
        trained_model = trainer.train(epochs=100)

        # 검증셋 기반 최적 임계값 탐색
        best_threshold = find_optimal_threshold(trained_model, val_loader, device)

        # 성과 측정
        evaluate_model(trained_model, test_loader, device, threshold=best_threshold)

        # 최종 실전 투입용 모델 파일 저장
        os.makedirs("models", exist_ok=True)
        save_path = "models/best_lstm_btc_5m.pth"
        torch.save(trained_model.state_dict(), save_path)
        print(f"\n[INFO] 💾 최고 성능의 모델 가중치가 안전하게 저장되었습니다: {save_path}")