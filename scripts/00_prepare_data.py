"""
scripts/00_prepare_data.py

Raw 5m OHLCV → 기술적 지표 계산 + 시장 국면(market_phase) 주석
+ train-only MinMaxScaler 정규화 → processed CSV + scaler 저장.

시장 국면 정의
  0 = Accumulation  2023-05-01 ~ 2023-10-15  (박스권 횡보)
  1 = Bull Run      2023-10-16 ~ 2025-06-30  (대세 상승장)
  2 = Bear / Crash  2025-07-01 ~             (대세 하락장)

train 정규화 기준: Phase 0 + Phase 1 (~ 2025-06-30)
val  기준: 2025-07-01 ~ 2025-10-31
test 기준: 2025-11-01 ~

사용법:
  python scripts/00_prepare_data.py
  python scripts/00_prepare_data.py --symbols BTC_USDT ETH_USDT
"""

import os
import sys
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.constants import PHASE_ACCUM_END, PHASE_BULL_END, PHASE_VAL_END

SYMBOLS = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'XRP_USDT']

# MinMaxScaler 적용 대상 컬럼
# - OHLCV: 학습 코드에서 raw pct_change로 재계산하므로 정규화 불필요
# - RSI/Stoch: 고정 범위(0~100) → /100 처리 (look-ahead 없음)
# - TD Sequential: 이미 /9 완료 → [0,1] 고정
MINMAX_COLS = [
    'macd', 'macd_signal', 'macd_hist',
    'atr',
    'ema_20', 'ema_50', 'ema_200',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
]

FIXED_SCALE_COLS = {
    'rsi':     100.0,
    'stoch_k': 100.0,
    'stoch_d': 100.0,
}


def _td_setup(close: np.ndarray, bullish: bool) -> np.ndarray:
    """TD Sequential Setup 카운터(0~9)를 /9로 정규화하여 반환."""
    n = len(close)
    out = np.zeros(n, dtype=np.float32)
    count = 0
    for i in range(4, n):
        cond = close[i] < close[i - 4] if bullish else close[i] > close[i - 4]
        count = min(count + 1, 9) if cond else 0
        out[i] = count / 9.0
    return out


def prepare_symbol(symbol: str, raw_dir: str, out_dir: str, scaler_dir: str) -> None:
    """단일 심볼의 raw CSV → processed CSV + scaler."""
    raw_path = os.path.join(raw_dir, f'{symbol}_5m_raw.csv')
    print(f'[{symbol}] raw 로드: {raw_path}')

    df_r = pd.read_csv(raw_path)
    df_r = df_r.sort_values('timestamp').reset_index(drop=True)

    o  = df_r['open'].values.astype(np.float64)
    h  = df_r['high'].values.astype(np.float64)
    l  = df_r['low'].values.astype(np.float64)
    c  = df_r['close'].values.astype(np.float64)
    v  = df_r['volume'].values.astype(np.float64)
    ts = df_r['timestamp'].values

    # ── 기술적 지표 계산 (raw 가격 기준) ──────────────────────────────────
    macd, macd_sig, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    rsi                       = talib.RSI(c, timeperiod=14)
    atr                       = talib.ATR(h, l, c, timeperiod=14)
    ema_20                    = talib.EMA(c, timeperiod=20)
    ema_50                    = talib.EMA(c, timeperiod=50)
    ema_200                   = talib.EMA(c, timeperiod=200)
    bb_up, bb_mid, bb_lo      = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    stoch_k, stoch_d          = talib.STOCH(h, l, c,
                                             fastk_period=14, slowk_period=3,
                                             slowk_matype=0, slowd_period=3, slowd_matype=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        bb_width = np.where(bb_mid > 1e-10, (bb_up - bb_lo) / bb_mid, 0.0)
        bb_pct   = np.where((bb_up - bb_lo) > 1e-10,
                             (c - bb_lo) / (bb_up - bb_lo), 0.5)

    td_buy  = _td_setup(c, bullish=True)
    td_sell = _td_setup(c, bullish=False)

    # ── DataFrame 조립 ────────────────────────────────────────────────────
    dt = pd.to_datetime(ts, unit='ms', utc=True).tz_convert(None)

    df = pd.DataFrame({
        'datetime'     : dt.strftime('%Y-%m-%d %H:%M:%S'),
        'timestamp'    : ts,
        'open'         : o,
        'high'         : h,
        'low'          : l,
        'close'        : c,
        'volume'       : v,
        'macd'         : macd,
        'macd_signal'  : macd_sig,
        'macd_hist'    : macd_hist,
        'rsi'          : rsi,
        'atr'          : atr,
        'ema_20'       : ema_20,
        'ema_50'       : ema_50,
        'ema_200'      : ema_200,
        'bb_upper'     : bb_up,
        'bb_middle'    : bb_mid,
        'bb_lower'     : bb_lo,
        'bb_width'     : bb_width,
        'bb_pct'       : bb_pct,
        'stoch_k'      : stoch_k,
        'stoch_d'      : stoch_d,
        'td_buy_setup' : td_buy,
        'td_sell_setup': td_sell,
    })

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 시장 국면 부여 ──────────────────────────────────────────────────────
    dt_s = pd.to_datetime(df['datetime'])
    df['market_phase'] = 2  # 기본: Bear
    df.loc[dt_s <= PHASE_ACCUM_END, 'market_phase'] = 0
    df.loc[(dt_s > PHASE_ACCUM_END) & (dt_s <= PHASE_BULL_END), 'market_phase'] = 1

    # ── 정규화 ──────────────────────────────────────────────────────────────
    # 고정 범위 (RSI, Stochastic): /100 → [0, 1]
    for col, div in FIXED_SCALE_COLS.items():
        df[col] = (df[col] / div).clip(0.0, 1.0)

    # TD Sequential: 이미 /9 완료, 범위 보장
    df['td_buy_setup']  = df['td_buy_setup'].clip(0.0, 1.0)
    df['td_sell_setup'] = df['td_sell_setup'].clip(0.0, 1.0)

    # MinMaxScaler: Phase 0+1 (train 기간)에만 fit → 미래 데이터 look-ahead 방지
    train_mask = dt_s <= PHASE_BULL_END
    df_train   = df.loc[train_mask, MINMAX_COLS]
    if len(df_train) == 0:
        raise RuntimeError(f'[{symbol}] train 데이터가 없습니다. raw 파일을 확인하세요.')

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(df_train.values)
    df[MINMAX_COLS] = np.clip(scaler.transform(df[MINMAX_COLS].values), 0.0, 1.0)

    # ── 저장 ────────────────────────────────────────────────────────────────
    os.makedirs(out_dir,    exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)

    out_path    = os.path.join(out_dir,    f'{symbol}_processed.csv')
    scaler_path = os.path.join(scaler_dir, f'{symbol}_scaler.pkl')

    df.to_csv(out_path, index=False)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # ── 리포트 ──────────────────────────────────────────────────────────────
    n_train = int(train_mask.sum())
    n_val   = int(((dt_s > PHASE_BULL_END) & (dt_s <= PHASE_VAL_END)).sum())
    n_test  = int((dt_s > PHASE_VAL_END).sum())
    phase_cnt = df['market_phase'].value_counts().sort_index().to_dict()

    print(f'  rows : {len(df):,}  '
          f'(train={n_train:,} [{n_train/len(df)*100:.1f}%], '
          f'val={n_val:,} [{n_val/len(df)*100:.1f}%], '
          f'test={n_test:,} [{n_test/len(df)*100:.1f}%])')
    print(f'  phase: 0(accum)={phase_cnt.get(0,0):,}  '
          f'1(bull)={phase_cnt.get(1,0):,}  '
          f'2(bear)={phase_cnt.get(2,0):,}')
    print(f'  CSV    → {out_path}')
    print(f'  scaler → {scaler_path}')

    # 지표 값 범위 검증
    out_of_range = (df[MINMAX_COLS + list(FIXED_SCALE_COLS.keys())] < 0.0).any().sum() + \
                   (df[MINMAX_COLS + list(FIXED_SCALE_COLS.keys())] > 1.0).any().sum()
    nan_count = df.isnull().sum().sum()
    print(f'  검증: NaN={nan_count}, 범위 이탈 컬럼={out_of_range}')
    if out_of_range > 0:
        print('  [WARN] 일부 컬럼이 [0,1] 범위를 벗어났습니다. 클리핑 확인 필요.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Raw OHLCV → processed CSV (phase-annotated, train-only MinMaxScaler)')
    parser.add_argument('--symbols',    nargs='+', default=SYMBOLS,
                        help='처리할 심볼 목록 (기본: 전체 4개 코인)')
    parser.add_argument('--raw-dir',    default=os.path.join(ROOT_DIR, 'data', 'raw'),
                        help='raw CSV 디렉터리')
    parser.add_argument('--out-dir',    default=os.path.join(ROOT_DIR, 'data', 'processed'),
                        help='processed CSV 출력 디렉터리')
    parser.add_argument('--scaler-dir', default=os.path.join(ROOT_DIR, 'data', 'processed', 'scalers'),
                        help='scaler pickle 출력 디렉터리')
    args = parser.parse_args()

    for sym in args.symbols:
        print(f'\n=== {sym} ===')
        prepare_symbol(sym, args.raw_dir, args.out_dir, args.scaler_dir)

    print('\n✅ 전처리 완료 — 다음 단계: python scripts/01_train_base.py')
