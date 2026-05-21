import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import talib
import warnings
warnings.filterwarnings('ignore')

from src.utils.platform_utils import get_optimal_workers, get_pin_memory
from src.utils.constants import PHASE_BULL_END, PHASE_VAL_END, LONG_TRAIN_END, LONG_VAL_END

# 하위 호환 별칭 (기존 코드가 _ prefix 를 사용하므로 유지)
_PHASE_BULL_END = PHASE_BULL_END
_PHASE_VAL_END  = PHASE_VAL_END

# ── LONG 전용 분할 경계 (현재 미적용 — 현재 best AUC 0.5411은 공통 분할로 달성) ────
# 향후 LONG 도메인 시프트(Train=Bull / Val=Bear) 개선 시도용 예비 상수.
# 적용 시 prepare_expert_data에서 expert_type=='long' 분기 추가 필요.
# from src.utils.constants import LONG_TRAIN_END  # TODO #8 활성화 예정

# ── 훈련 데이터 언더샘플링 상한 (Pos 1 : Neg 최대 N) ─────────────────────────
# BCEWithLogitsLoss pos_weight 계산과 일치시켜 이중 보정을 방지합니다.
_MAX_NEG_RATIO = 3

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
        # [Fix 1] end_idx 캔들 자체를 피처에 포함 (off-by-one 수정)
        start_idx = end_idx - self.seq_length + 1
        
        x = self.features[start_idx : end_idx + 1]
        y = self.targets[end_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ─────────────────────────────────────────────────────────────
# 2. 전문가 신경망 구조 (원본 유지)
# ─────────────────────────────────────────────────────────────
class PriceActionExpert(nn.Module):
    """
    가격/거래량(OHLCV) + 기술적 지표 통합 전문가.
    입력 피처 수: price_vol(5) + context(18) = 23개 (input_dim으로 동적 결정)
    출력: logit (BCEWithLogitsLoss 와 함께 사용)
    """
    def __init__(self, input_dim=5, hidden_dim=64, dropout=0.3, use_attention=False):
        super(PriceActionExpert, self).__init__()
        # [Fix 9] input_dim 동적 파라미터화 (OHLCV 5 → OHLCV+context 23)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        # [Fix 5] 시계열에 적합한 LayerNorm으로 교체 (BatchNorm1d 제거)
        self.ln = nn.LayerNorm(hidden_dim)
        # [Fix 12] use_attention=True 일 때만 attention 레이어 생성 (SHORT: True, LONG: False)
        # SHORT는 Bear 기간 패턴 포착에 attention 효과적; LONG은 마지막 hidden state가 최적
        self.attn = nn.Linear(hidden_dim, 1, bias=False) if use_attention else None
        # [Fix 10] FC 중간층 = hidden_dim // 2 (hidden_dim 변경 시 자동 스케일)
        _fc_dim = hidden_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, _fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(_fc_dim, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)                                          # [B, T, H]
        if self.attn is not None:
            # [Fix 11/12] softmax attention: SHORT에서 특정 시점 패턴 집중
            attn_w = torch.softmax(self.attn(out).squeeze(-1), dim=1) # [B, T]
            feat = self.ln((attn_w.unsqueeze(-1) * out).sum(dim=1))   # [B, H]
        else:
            # LONG: 마지막 hidden state 사용 (Fix 10 방식, Bull→Bear 일반화 최적)
            feat = self.ln(out[:, -1, :])                              # [B, H]
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
        # [Fix 5] 시계열에 적합한 LayerNorm으로 교체 (BatchNorm1d 제거)
        self.ln = nn.LayerNorm(hidden_dim)
        # [Fix 6] Sigmoid 제거 → BCEWithLogitsLoss 와 함께 사용 (수치 안정성 향상)
        # [Fix 10] FC 중간층 = hidden_dim // 2 (hidden_dim 변경 시 자동 스케일)
        _fc_dim = hidden_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, _fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(_fc_dim, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = self.ln(out[:, -1, :])
        return self.fc(feat).squeeze(1)


# ─────────────────────────────────────────────────────────────
# 3. 데이터 파이프라인 (시계열 파괴 버그 및 Leakage 해결)
# ─────────────────────────────────────────────────────────────
def prepare_expert_data(filepath, expert_type, seq_length=120, long_split=False):
    """
    expert_type별 학습 데이터를 구성합니다.
    - long   : 롱 TP 우선 도달 시 1
    - short  : 숏 TP 우선 도달 시 1
    - context: 롱/숏 어느 방향이든 변동성 이벤트 발생 시 1

    시계열 분할은 인덱스 기반으로 수행하며, split 경계에는 seq_length 간격을 둬
    윈도우 겹침에 의한 누출 가능성을 줄입니다.

    Args:
        long_split (bool): Long 전문가 전용 분할 전략 적용 여부.
            True  → train = Phase 0+1 초중반 (~LONG_TRAIN_END),
                    val   = Phase 1 후반 (LONG_TRAIN_END+1step ~ LONG_VAL_END)
                    → Train/Val 모두 Bull 레짐, 도메인 시프트 완화
            False → 기본 분할 (train = Phase 0+1, val = Phase 2 초반)
            expert_type != 'long' 일 때는 무시됩니다.
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
    # quantile 경계: Phase 0+1 (train 기간 = ~ 2025-06-30)만으로 산출해 look-ahead 방지.
    price_cols = ['open', 'high', 'low', 'close']
    _dt_for_cutoff = pd.to_datetime(df['datetime'])
    train_cutoff = int((_dt_for_cutoff <= _PHASE_BULL_END).sum())
    if train_cutoff == 0:  # fallback: 전체 raw 데이터가 bear 기간만 있는 경우
        train_cutoff = int(len(df) * 0.70)
    # [Fix 2] 모든 OHLC를 직전 캔들의 종가(prev_close) 기준 Log Return으로 통일
    # → 캔들 내부 구조(O/H/L/C 상대 위치)가 보존되어 캔들 형태가 유지됨
    prev_close = df['close_raw'].shift(1)
    for col in price_cols:
        df[col] = np.log(df[f'{col}_raw'] / prev_close).fillna(0)
        q_lo = df[col].iloc[:train_cutoff].quantile(0.001)
        q_hi = df[col].iloc[:train_cutoff].quantile(0.999)
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
    
    # ── 정답지(Label) 생성 로직 (ATR 기반 동적 임계치) ──
    horizon = 72
    TP_MULT = 2.0   # ATR 배수: TP = 2.0 × ATR%
    SL_MULT = 1.0   # ATR 배수: SL = 1.0 × ATR%
    MIN_TP  = 0.5   # 최소 TP (%) — 변동성 극저점 방어
    MIN_SL  = 0.25  # 최소 SL (%) — 변동성 극저점 방어

    # 14주기 ATR 계산 및 가격 대비 비율(%)로 변환
    # h, l, c 는 위 캔들 패턴 블록에서 이미 df['*_raw'].values 로 정의됨
    close_prices = df['close_raw'].values
    high_prices  = df['high_raw'].values   # [Fix 3] 꼬리 기반 라벨링용
    low_prices   = df['low_raw'].values    # [Fix 3] 꼬리 기반 라벨링용
    n = len(close_prices)
    _atr_abs = talib.ATR(h, l, close_prices, timeperiod=14)
    _atr_pct  = np.where(close_prices > 0,
                         _atr_abs / close_prices * 100.0,
                         np.nan)
    # NaN(초기 14봉 warm-up 구간) → 전체 중앙값으로 fallback
    _atr_median = float(np.nanmedian(_atr_pct))
    _atr_pct = np.where(np.isnan(_atr_pct), _atr_median, _atr_pct)

    targets = np.zeros(n, dtype=np.float32)

    for i in range(n - horizon):
        # 시점별 동적 임계치
        tp_thresh = max(MIN_TP, TP_MULT * _atr_pct[i])
        sl_thresh = max(MIN_SL, SL_MULT * _atr_pct[i])

        curr_p = close_prices[i]
        # [Fix 3] 미래 윈도우의 고가·저가로 TP/SL 판별 (꼬리 노이즈 반영)
        future_high = high_prices[i+1: i+1+horizon]
        future_low  = low_prices[i+1:  i+1+horizon]
        ret_high = (future_high - curr_p) / curr_p * 100
        ret_low  = (future_low  - curr_p) / curr_p * 100

        # Long: 고가로 익절 / 저가로 손절
        hit_tp_long = np.where(ret_high >= tp_thresh)[0]
        hit_sl_long = np.where(ret_low  <= -sl_thresh)[0]
        idx_tp_long = hit_tp_long[0] if len(hit_tp_long) > 0 else horizon + 1
        idx_sl_long = hit_sl_long[0] if len(hit_sl_long) > 0 else horizon + 1
        is_long = (idx_tp_long < idx_sl_long) and (idx_tp_long <= horizon)

        # Short: 저가로 익절 / 고가로 손절
        hit_tp_short = np.where(ret_low  <= -tp_thresh)[0]
        hit_sl_short = np.where(ret_high >= sl_thresh)[0]
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

    # ── 상대적/가격정규화 지표 피처 생성 (raw 컬럼 제거 전) ────────────────────
    # 절대 가격 수준 종속성을 제거해 Bear/Bull 기간 모두 일관된 값 범위를 보장합니다.
    _cr      = df['close_raw'].values.astype(np.float64)
    _safe_cr = np.where(_cr > 0, _cr, 1.0)  # 0-division 방지

    # EMA 거리 비율: (ema / close - 1)  → 양수=가격이 EMA 위, 음수=아래
    _ema20_v  = talib.EMA(_cr, timeperiod=20)
    _ema50_v  = talib.EMA(_cr, timeperiod=50)
    _ema200_v = talib.EMA(_cr, timeperiod=200)
    df['ema_20_r']  = np.clip(np.where(np.isnan(_ema20_v),  0.0, _ema20_v  / _safe_cr - 1.0), -0.30, 0.30)
    df['ema_50_r']  = np.clip(np.where(np.isnan(_ema50_v),  0.0, _ema50_v  / _safe_cr - 1.0), -0.50, 0.50)
    df['ema_200_r'] = np.clip(np.where(np.isnan(_ema200_v), 0.0, _ema200_v / _safe_cr - 1.0), -1.00, 1.00)

    # MACD 가격 정규화: macd / close → 가격 수준과 무관한 모멘텀 강도
    _macd_v, _macd_s_v, _macd_h_v = talib.MACD(_cr, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_r']        = np.clip(np.where(np.isnan(_macd_v),   0.0, _macd_v   / _safe_cr), -0.05, 0.05)
    df['macd_signal_r'] = np.clip(np.where(np.isnan(_macd_s_v), 0.0, _macd_s_v / _safe_cr), -0.05, 0.05)
    df['macd_hist_r']   = np.clip(np.where(np.isnan(_macd_h_v), 0.0, _macd_h_v / _safe_cr), -0.03, 0.03)

    # 불필요 원본 제거
    drop_cols = [c for c in df.columns if c.endswith('_raw')]
    df.drop(columns=drop_cols, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)  # dropna 후 인덱스 정합성 보장

    price_vol_cols = price_cols + vol_col
    # 절대 가격 수준 컬럼: 상대 버전(ema_*_r, macd_*_r)으로 대체되었으므로 제외
    # bb_upper/middle/lower 도 절대 가격 → bb_pct/bb_width(상대 비율)만 유지
    _abs_price_feat = [
        'ema_20', 'ema_50', 'ema_200',
        'bb_upper', 'bb_middle', 'bb_lower',
        'macd', 'macd_signal', 'macd_hist',
    ]
    exclude_cols = (
        ['timestamp', 'datetime', 'Target', '1h_ema_50', '1h_ema_200', 'market_phase']
        + _abs_price_feat
    )
    context_cols = [c for c in df.columns if c not in price_vol_cols + exclude_cols]

    if expert_type in ['long', 'short']:
        # [Fix 9] OHLCV(5) + 기술적 지표(18) 통합 피처: 방향성 예측에 필요한 컨텍스트 추가
        # context_cols: rsi, bb_width/pct, stoch_k/d, td_setup, ema_*_r, macd_*_r, pat_*
        features = df[price_vol_cols + context_cols].values.astype(np.float32)
    else:
        features = df[context_cols].values.astype(np.float32)
        
    targets = df['Target'].values.astype(np.float32)

    # ── 시장 국면 기반 인덱스 분할 ──────────────────────────────────────────
    # 기본 분할:
    #   Phase 0+1 (~ 2025-06-30) → train
    #   Phase 2 early (2025-07-01 ~ 2025-10-31) → val
    #   Phase 2 late  (2025-11-01 ~)             → test
    # Long 전용 분할 (long_split=True):
    #   Phase 0+1 초중반 (~ LONG_TRAIN_END=2024-06-30) → train
    #   Phase 1 후반 (2024-07-01 ~ LONG_VAL_END=2025-06-30) → val
    #   Phase 2 ~ → test (기본과 동일)
    # 경계에서 seq_length 간격을 둬 윈도우 겹침에 의한 누출을 방지합니다.
    _dt = pd.to_datetime(df['datetime'])
    _bull_end_pos = int((_dt <= _PHASE_BULL_END).sum())
    _val_end_pos  = int((_dt <= _PHASE_VAL_END).sum())

    # [Fix 1] off-by-one 수정: end_idx = seq_length - 1 일 때 start_idx = 0
    total_valid_indices = np.arange(seq_length - 1, len(features))

    if long_split and expert_type == 'long':
        _long_train_end_pos = int((_dt <= LONG_TRAIN_END).sum())
        _long_val_end_pos   = int((_dt <= LONG_VAL_END).sum())
        print(
            f"[Long 전용 분할] train: ~{LONG_TRAIN_END.date()} ({_long_train_end_pos} rows), "
            f"val: ~{LONG_VAL_END.date()} ({_long_val_end_pos - _long_train_end_pos} rows)"
        )
        raw_train_indices = total_valid_indices[
            total_valid_indices < _long_train_end_pos - horizon
        ]
        raw_val_indices   = total_valid_indices[
            (total_valid_indices >= _long_train_end_pos + seq_length + horizon) &
            (total_valid_indices <  _long_val_end_pos - horizon)
        ]
        raw_test_indices  = total_valid_indices[
            total_valid_indices >= _val_end_pos + seq_length + horizon
        ]
    else:
        raw_train_indices = total_valid_indices[
            total_valid_indices < _bull_end_pos - horizon
        ]
        raw_val_indices   = total_valid_indices[
            (total_valid_indices >= _bull_end_pos + seq_length + horizon) &
            (total_valid_indices <  _val_end_pos - horizon)
        ]
        raw_test_indices  = total_valid_indices[
            total_valid_indices >= _val_end_pos + seq_length + horizon
        ]

    # 언더샘플링 헬퍼: 강제 1:1 제거 → 최대 1:_MAX_NEG_RATIO 비율로 제한
    # BCEWithLogitsLoss pos_weight 와 동일 비율을 사용해 이중 보정을 방지합니다.
    def get_capped_indices(indices, target_array):
        sub_targets = target_array[indices]
        pos_idx = indices[np.where(sub_targets == 1.0)[0]]
        neg_idx = indices[np.where(sub_targets == 0.0)[0]]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return indices  # 한쪽 클래스가 비어 있으면 원본 그대로

        max_neg = len(pos_idx) * _MAX_NEG_RATIO
        if len(neg_idx) > max_neg:
            rng = np.random.default_rng(42)
            neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)

        return np.sort(np.concatenate([pos_idx, neg_idx]))

    train_indices = get_capped_indices(raw_train_indices, targets)
    # [Fix 4] Val/Test는 평가 무결성을 위해 언더샘플링 없이 원본 인덱스 그대로 사용
    val_indices   = raw_val_indices
    test_indices  = raw_test_indices

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

    # 훈련 데이터 양성 비율(다운샘플링 이전 raw 기준) → Focal Loss alpha 계산에 사용
    _raw_train_targets = targets[raw_train_indices]
    _n_pos = float((_raw_train_targets == 1.0).sum())
    _n_neg = float((_raw_train_targets == 0.0).sum())
    pos_weight_raw = _n_neg / (_n_pos + 1e-8)  # neg/pos 비율; 1보다 크면 양성 희소
    _val_targets  = targets[val_indices]
    _val_pos_pct  = float((_val_targets == 1.0).mean()) * 100 if len(_val_targets) > 0 else 0.0
    print(
        f"[INFO] 🎯 시계열 유지 분할 완료 "
        f"(ATR 동적 임계치 | max_neg_ratio=1:{_MAX_NEG_RATIO} | log_return) "
        f"(Train: {len(train_indices):,}, Val: {len(val_indices):,}) "
        f"| raw pos_weight={pos_weight_raw:.2f} "
        f"(pos={int(_n_pos):,} neg={int(_n_neg):,}) "
        f"| Val 양성={_val_pos_pct:.1f}% "
        f"({int((_val_targets==1.0).sum()):,}/{len(val_indices):,})"
    )

    return train_loader, val_loader, test_loader, features.shape[1], pos_weight_raw