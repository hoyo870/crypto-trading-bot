import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import talib
import warnings
warnings.filterwarnings('ignore')

from src.utils.platform_utils import get_optimal_workers, get_pin_memory

# ─────────────────────────────────────────────────────────────
# 1. 시계열 커스텀 데이터셋 (연속성 보장 구조로 개선)
# ─────────────────────────────────────────────────────────────
class CryptoExpertDataset(Dataset):
    def __init__(self, features, targets, seq_length, valid_indices):
        """
        시계열 연속성을 유지하기 위해 전체 배열을 보관하고,
        target 시점 인덱스(valid_indices)로만 샘플을 구성합니다.
        """
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # valid_indices에서 실제 데이터의 끝점(target 시점)을 가져옴
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_length
        
        x = self.features[start_idx : end_idx]
        y = self.targets[end_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ─────────────────────────────────────────────────────────────
# 2. 전문가 신경망 구조 (원본 유지)
# ─────────────────────────────────────────────────────────────
class PriceActionExpert(nn.Module):
    """
    가격/거래량(OHLCV) 전용 전문가.
    입력 피처 수: 5개 (Open, High, Low, Close, Volume)
    출력: 0~1 확률값 (Sigmoid)
    """
    def __init__(self, hidden_dim=64, dropout=0.3):
        super(PriceActionExpert, self).__init__()
        # 입력 차원: 5 (Open, High, Low, Close, Volume)
        self.lstm = nn.LSTM(5, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
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

class ContextExpert(nn.Module):
    """
    보조지표/캔들패턴 전용 전문가.
    입력에서 가격/거래량을 제외한 컨텍스트 피처를 사용하고,
    출력은 0~1 시장 에너지 점수입니다.
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
# 3. 데이터 파이프라인 (시계열 파괴 버그 및 Leakage 해결)
# ─────────────────────────────────────────────────────────────
def prepare_expert_data(filepath, expert_type, seq_length=120):
    """
    expert_type별 학습 데이터를 구성합니다.
    - long   : 롱 TP 우선 도달 시 1
    - short  : 숏 TP 우선 도달 시 1
    - context: 롱/숏 어느 방향이든 변동성 이벤트 발생 시 1

    시계열 분할은 인덱스 기반으로 수행하며, split 경계에는 seq_length 간격을 둬
    윈도우 겹침에 의한 누출 가능성을 줄입니다.
    """
    print(f"[INFO] {expert_type.upper()} 전문가용 데이터 로드 및 전처리 중...")
    df = pd.read_csv(filepath)

    if 'atr' in df.columns:
        df.drop(columns=['atr'], inplace=True)

    raw_filepath = filepath.replace(
        os.path.join("data", "processed"),
        os.path.join("data", "raw")
    ).replace("_processed.csv", "_5m_raw.csv")
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

    # 캔들 패턴 생성
    o, h, l, c = df['open_raw'].values, df['high_raw'].values, df['low_raw'].values, df['close_raw'].values
    df['pat_doji'] = talib.CDLDOJI(o, h, l, c) / 100.0
    df['pat_hammer'] = talib.CDLHAMMER(o, h, l, c) / 100.0
    df['pat_engulfing'] = talib.CDLENGULFING(o, h, l, c) / 100.0
    df['pat_morningstar'] = talib.CDLMORNINGSTAR(o, h, l, c) / 100.0
    df['pat_eveningstar'] = talib.CDLEVENINGSTAR(o, h, l, c) / 100.0
    
    # ── 정답지(Label) 생성 로직 ──
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
            targets[i] = 1.0 if is_long or is_short else 0.0

    df['Target'] = targets

    # 불필요 원본 제거
    drop_cols = [c for c in df.columns if c.endswith('_raw')]
    df.drop(columns=drop_cols, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    price_vol_cols = price_cols + vol_col
    exclude_cols = ['timestamp', 'datetime', 'Target', '1h_ema_50', '1h_ema_200']
    context_cols = [c for c in df.columns if c not in price_vol_cols + exclude_cols]

    if expert_type in ['long', 'short']:
        features = df[price_vol_cols].values.astype(np.float32)
    else:
        features = df[context_cols].values.astype(np.float32)
        
    targets = df['Target'].values.astype(np.float32)

    # ── 시계열을 유지하는 인덱스 기반 분할/다운샘플링 ──
    # LSTM 생성을 위해 유효한 최소 인덱스는 seq_length 부터 시작합니다.
    total_valid_indices = np.arange(seq_length, len(features))
    
    train_end_idx = int(len(total_valid_indices) * 0.70)
    val_end_idx = int(len(total_valid_indices) * 0.85)

    raw_train_indices = total_valid_indices[:train_end_idx]
    # Train/Validation/Test 경계에서 윈도우 중첩을 피하기 위해 seq_length 간격을 둡니다.
    raw_val_indices = total_valid_indices[train_end_idx + seq_length : val_end_idx]
    raw_test_indices = total_valid_indices[val_end_idx + seq_length :]

    # 다운샘플링 헬퍼 함수
    def get_balanced_indices(indices, target_array):
        sub_targets = target_array[indices]
        pos_idx = indices[np.where(sub_targets == 1.0)[0]]
        neg_idx = indices[np.where(sub_targets == 0.0)[0]]
        
        min_len = min(len(pos_idx), len(neg_idx))
        if min_len == 0:
            return indices  # 한쪽 클래스가 비어 있으면 원본 인덱스를 그대로 사용
            
        rng = np.random.default_rng(42)
        pos_idx_sampled = rng.choice(pos_idx, size=min_len, replace=False)
        neg_idx_sampled = rng.choice(neg_idx, size=min_len, replace=False)
        
        balanced_indices = np.sort(np.concatenate([pos_idx_sampled, neg_idx_sampled]))
        return balanced_indices

    # 현재 구현은 Train/Val/Test 모두 동일한 1:1 밸런싱 규칙을 적용합니다.
    train_indices = get_balanced_indices(raw_train_indices, targets)
    val_indices = get_balanced_indices(raw_val_indices, targets)
    test_indices = get_balanced_indices(raw_test_indices, targets)

    # 전체 배열(features, targets)은 자르지 않고 그대로 Dataset에 넘김 (인덱스만 전달)
    train_dataset = CryptoExpertDataset(features, targets, seq_length, train_indices)
    val_dataset = CryptoExpertDataset(features, targets, seq_length, val_indices)
    test_dataset = CryptoExpertDataset(features, targets, seq_length, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=get_optimal_workers(),
                              pin_memory=get_pin_memory(),
                              persistent_workers=(get_optimal_workers() > 0))
    val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False,
                              num_workers=get_optimal_workers(),
                              pin_memory=get_pin_memory(),
                              persistent_workers=(get_optimal_workers() > 0))
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False,
                              num_workers=get_optimal_workers(),
                              pin_memory=get_pin_memory(),
                              persistent_workers=(get_optimal_workers() > 0))

    print(f"[INFO] 🎯 시계열 유지 다운샘플링 및 분할 완료 (Train: {len(train_indices):,}, Val: {len(val_indices):,})")
    
    return train_loader, val_loader, test_loader, features.shape[1]