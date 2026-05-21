"""
세 가지 TP/SL 설정별 레이블 양성 비율 분석
기간: 2023-05-05 ~ 2026-05-04
"""
import numpy as np
import pandas as pd
import talib

FILES = {
    'BTC': 'data/raw/BTC_USDT_5m_raw.csv',
    'ETH': 'data/raw/ETH_USDT_5m_raw.csv',
    'SOL': 'data/raw/SOL_USDT_5m_raw.csv',
    'XRP': 'data/raw/XRP_USDT_5m_raw.csv',
}

DATE_START = '2023-05-05'
DATE_END   = '2026-05-04'

HORIZONS = {
    '6h':  72,
    '12h': 144,
    '24h': 288,
}
TP_MULT = 2.0
SL_MULT = 1.0

CONFIGS = {
    '기존(0.5/0.25)':  (0.5,  0.25),
    '개선(0.15/0.07)': (0.15, 0.07),
    '체크(1.0/0.5)':   (1.0,  0.5),
}


def compute_labels_vectorized(high, low, close, tp_pct_arr, sl_pct_arr, horizon):
    """
    numpy 벡터화 레이블 계산.
    tp_pct_arr, sl_pct_arr: 각 봉별 동적 임계치 (%, 1D array)
    반환: long_labels, short_labels (0/1 array)
    """
    n = len(close)
    # (n, horizon) 미래 윈도우 구성
    # i봉 기준 미래 [i+1, i+1+horizon) 고가/저가
    idx = np.arange(n)[:, None] + np.arange(1, horizon + 1)[None, :]  # (n, horizon)
    idx = np.clip(idx, 0, n - 1)

    fut_high = high[idx]   # (n, horizon)
    fut_low  = low[idx]    # (n, horizon)

    # 수익률 (%)
    c = close[:, None]     # (n, 1)
    ret_high = (fut_high - c) / c * 100  # (n, horizon)
    ret_low  = (fut_low  - c) / c * 100  # (n, horizon)

    tp = tp_pct_arr[:, None]   # (n, 1)
    sl = sl_pct_arr[:, None]   # (n, 1)

    # TP/SL 첫 도달 시점 (horizon+1 = 미도달)
    H1 = horizon + 1

    # Long: 고가 TP / 저가 SL
    tp_long_hit  = (ret_high >= tp)
    sl_long_hit  = (ret_low  <= -sl)
    # 첫 True 인덱스: argmax는 True 없으면 0 반환 → 보정
    idx_tp_long = np.where(tp_long_hit.any(axis=1), tp_long_hit.argmax(axis=1), H1)
    idx_sl_long = np.where(sl_long_hit.any(axis=1), sl_long_hit.argmax(axis=1), H1)
    is_long = (idx_tp_long < idx_sl_long) & (idx_tp_long <= horizon)

    # Short: 저가 TP / 고가 SL
    tp_short_hit = (ret_low  <= -tp)
    sl_short_hit = (ret_high >= sl)
    idx_tp_short = np.where(tp_short_hit.any(axis=1), tp_short_hit.argmax(axis=1), H1)
    idx_sl_short = np.where(sl_short_hit.any(axis=1), sl_short_hit.argmax(axis=1), H1)
    is_short = (idx_tp_short < idx_sl_short) & (idx_tp_short <= horizon)

    # Long 레이블: long 단독 or (long+short 동시일 때 long TP 먼저)
    long_lbl = (is_long & ~is_short) | (is_long & is_short & (idx_tp_long < idx_tp_short))
    # Short 레이블
    short_lbl = (is_short & ~is_long) | (is_long & is_short & (idx_tp_short < idx_tp_long))
    # Context 레이블
    ctx_lbl = is_long | is_short

    return long_lbl.astype(float), short_lbl.astype(float), ctx_lbl.astype(float)


print(f"\n{'='*90}")
print(f" 레이블 양성 비율 분석  ({DATE_START} ~ {DATE_END})")
print(f" TP=max(MIN, {TP_MULT}×ATR%) | SL=max(MIN, {SL_MULT}×ATR%)")
print(f"{'='*90}")

# sym → cfg → horizon → (long%, short%, ctx%)
all_results = {}

for sym, path in FILES.items():
    df = pd.read_csv(path, parse_dates=['datetime'])
    mask = (df['datetime'] >= DATE_START) & (df['datetime'] <= DATE_END)
    df = df[mask].reset_index(drop=True)
    if len(df) == 0:
        print(f"[WARN] {sym}: 해당 기간 데이터 없음")
        continue

    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(c)

    atr_abs = talib.ATR(h, l, c, timeperiod=14)
    atr_pct = np.where(c > 0, atr_abs / c * 100.0, np.nan)
    atr_median = float(np.nanmedian(atr_pct))
    atr_pct = np.where(np.isnan(atr_pct), atr_median, atr_pct)

    print(f"\n[{sym}]  총 {n:,}봉  ATR% 중앙값={atr_median:.4f}%")
    print(f"  {'설정':<20} {'horizon':>8} {'Long%':>7} {'Short%':>7} {'Context%':>9}")
    print(f"  {'-'*55}")

    sym_results = {}
    for cfg_name, (min_tp, min_sl) in CONFIGS.items():
        tp_arr = np.maximum(min_tp, TP_MULT * atr_pct)
        sl_arr = np.maximum(min_sl, SL_MULT * atr_pct)
        sym_results[cfg_name] = {}

        for h_label, horizon in HORIZONS.items():
            valid_n = n - horizon
            long_lbl, short_lbl, ctx_lbl = compute_labels_vectorized(
                h, l, c, tp_arr, sl_arr, horizon
            )
            long_r  = long_lbl[:valid_n].mean() * 100
            short_r = short_lbl[:valid_n].mean() * 100
            ctx_r   = ctx_lbl[:valid_n].mean() * 100
            sym_results[cfg_name][h_label] = (long_r, short_r, ctx_r)
            print(f"  {cfg_name:<20} {h_label:>8} {long_r:>6.2f}%  {short_r:>6.2f}%  {ctx_r:>8.2f}%")

        print(f"  {'-'*55}")

    all_results[sym] = sym_results

# 4종목 평균 요약
print(f"\n{'='*90}")
print(f" 4종목 평균")
print(f"  {'설정':<20} {'horizon':>8} {'Long%':>7} {'Short%':>7} {'Context%':>9}")
print(f"  {'-'*55}")
for cfg_name in CONFIGS:
    for h_label in HORIZONS:
        vals = [all_results[s][cfg_name][h_label] for s in all_results]
        avg_long  = np.mean([v[0] for v in vals])
        avg_short = np.mean([v[1] for v in vals])
        avg_ctx   = np.mean([v[2] for v in vals])
        print(f"  {cfg_name:<20} {h_label:>8} {avg_long:>6.2f}%  {avg_short:>6.2f}%  {avg_ctx:>8.2f}%")
    print(f"  {'-'*55}")

print(f"{'='*90}\n")
