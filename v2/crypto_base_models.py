import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import talib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1. 시계열 커스텀 데이터셋 (3가지 목적을 위해 변형 가능하도록 설계)
# ─────────────────────────────────────────────────────────────
class CryptoExpertDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ─────────────────────────────────────────────────────────────
# 2. 전문가 신경망 구조
# ─────────────────────────────────────────────────────────────
class PriceActionExpert(nn.Module):
    """
    1차 모델 (Model A/B): 오직 순수 가격(OHLC)과 거래량(Volume)만 봅니다.
    입력 피처 수: 5개 (Open, High, Low, Close, Volume)
    출력: 0~1 사이의 단일 확률값 (Sigmoid)
    """
    def __init__(self, hidden_dim=64, dropout=0.3):
        super(PriceActionExpert, self).__init__()
        self.lstm = nn.LSTM(5, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0~1 확률 도출
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = self.bn(out[:, -1, :])
        return self.fc(feat).squeeze(1)

class ContextExpert(nn.Module):
    """
    2차 참모 (Model C): 가격을 제외한 보조지표와 캔들 패턴만 봅니다.
    출력: 0~1 사이의 단일 확률값 (Sigmoid) - 시장 에너지 점수
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(ContextExpert, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = self.bn(out[:, -1, :])
        return self.fc(feat).squeeze(1)

# ─────────────────────────────────────────────────────────────
# 3. 데이터 파이프라인 (이진 분류 라벨링으로 변경)
# ─────────────────────────────────────────────────────────────
def prepare_expert_data(filepath, expert_type, seq_length=120):
    """
    expert_type에 따라 피처를 자르고, 라벨(정답지)을 0/1로 쪼개서 반환합니다.
    - 'long'   : Model A (가격/거래량 피처, 롱 닿으면 1 아니면 0)
    - 'short'  : Model B (가격/거래량 피처, 숏 닿으면 1 아니면 0)
    - 'context': Model C (지표/패턴 피처, 변동성(롱or숏) 닿으면 1 아니면 0)
    """
    print(f"[INFO] {expert_type.upper()} 전문가용 데이터 로드 및 전처리 중...")
    df = pd.read_csv(filepath)

    if 'atr' in df.columns:
        df.drop(columns=['atr'], inplace=True)

    raw_filepath = filepath.replace("_processed.csv", "_5m_raw.csv")
    df_raw = pd.read_csv(raw_filepath)
    df = pd.merge(df, df_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']], on='timestamp', suffixes=('', '_raw'))

    # 피처 스케일링
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = df[f'{col}_raw'].pct_change().fillna(0)
        q_lo, q_hi = df[col].quantile(0.001), df[col].quantile(0.999)
        df[col] = df[col].clip(q_lo, q_hi)
    
    vol_ma = df['volume_raw'].rolling(24).mean() + 1e-9
    vol_col = ['volume']
    df[vol_col[0]] = (df['volume_raw'] / vol_ma).clip(0, 10) 

    # 캔들 패턴 (Context Expert를 위해)
    o, h, l, c = df['open_raw'].values, df['high_raw'].values, df['low_raw'].values, df['close_raw'].values
    df['pat_doji'] = talib.CDLDOJI(o, h, l, c) / 100.0
    df['pat_hammer'] = talib.CDLHAMMER(o, h, l, c) / 100.0
    df['pat_engulfing'] = talib.CDLENGULFING(o, h, l, c) / 100.0
    df['pat_morningstar'] = talib.CDLMORNINGSTAR(o, h, l, c) / 100.0
    df['pat_eveningstar'] = talib.CDLEVENINGSTAR(o, h, l, c) / 100.0
    
    # ── 정답지(Label) 생성: 전문가별 목표가 다르다 ──
    horizon = 72
    tp_thresh = 1.4  
    sl_thresh = 0.7  
    close_prices = df['close_raw'].values
    n = len(close_prices)
    
    targets = np.zeros(n, dtype=np.float32) 

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

        if expert_type == 'long':
            targets[i] = 1.0 if (is_long and not is_short) or (is_long and is_short and idx_tp_long < idx_tp_short) else 0.0
        elif expert_type == 'short':
            targets[i] = 1.0 if (is_short and not is_long) or (is_long and is_short and idx_tp_short < idx_tp_long) else 0.0
        elif expert_type == 'context':
            # Context 참모는 방향 모름. 위든 아래든 터지는(1) 자리인가 횡보(0)인가만 판단
            targets[i] = 1.0 if is_long or is_short else 0.0

    df['Target'] = targets

    # 불필요 원본 제거
    drop_cols = [c for c in df.columns if c.endswith('_raw')]
    df.drop(columns=drop_cols, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # ── 전문가별 입력 피처(Input) 분리 ──
    price_vol_cols = price_cols + vol_col
    exclude_cols = ['timestamp', 'datetime', 'Target', '1h_ema_50', '1h_ema_200']
    context_cols = [c for c in df.columns if c not in price_vol_cols + exclude_cols]

    if expert_type in ['long', 'short']:
        features = df[price_vol_cols].values.astype(np.float32)
    else: # context
        features = df[context_cols].values.astype(np.float32)
        
    targets = df['Target'].values.astype(np.float32)

    # 이진 분류를 위한 다운샘플링 (Positive와 Negative 비율 1:1)
    pos_idx = np.where(targets == 1.0)[0]
    neg_idx = np.where(targets == 0.0)[0]
    min_len = min(len(pos_idx), len(neg_idx))
    rng = np.random.default_rng(42)
    pos_idx = rng.choice(pos_idx, size=min_len, replace=False)
    neg_idx = rng.choice(neg_idx, size=min_len, replace=False)
    keep_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    
    features = features[keep_idx]
    targets = targets[keep_idx]

    train_end = int(len(features) * 0.70)
    val_end = int(len(features) * 0.85)

    X_train, y_train = features[:train_end], targets[:train_end]
    X_val, y_val = features[train_end:val_end], targets[train_end:val_end]
    X_test, y_test = features[val_end:], targets[val_end:]

    train_dataset = CryptoExpertDataset(X_train, y_train, seq_length)
    val_dataset = CryptoExpertDataset(X_val, y_val, seq_length)
    test_dataset = CryptoExpertDataset(X_test, y_test, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"[INFO] 🎯 1:1 다운샘플링 완료 (총 샘플: {len(keep_idx):,})")
    return train_loader, val_loader, test_loader, features.shape[1]