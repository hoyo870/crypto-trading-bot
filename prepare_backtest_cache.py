"""
=============================================================================
 백테스트 피처 캐시 생성 스크립트
=============================================================================
 실행 방법 : python prepare_backtest_cache.py
 실행 시점 : 학습 완료 후 백테스트 전 1회만 실행

 역할:
  - 백테스트가 매번 재실행하던 무거운 전처리(CSV 로딩, merge, 스케일링)를
    한 번만 수행하고 결과를 .npz + .json 캐시 파일로 저장합니다.
  - 이후 백테스트는 캐시를 즉시 로드해서 실행 속도가 대폭 빨라집니다.

 생성 파일:
  - data/{SYMBOL}_backtest_cache.npz  : 피처 행렬 / 원본 종가 / 타임스탬프
  - data/{SYMBOL}_backtest_meta.json  : 보조지표 컬럼 목록 / val_end 인덱스
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# 처리할 심볼 목록 (원하는 심볼만 추가/제거하세요)
# ─────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTC_USDT",
    "ETH_USDT",
    "SOL_USDT",
    "XRP_USDT",
]
DATA_DIR = "data"


def build_cache(symbol: str):
    """단일 심볼의 피처 엔지니어링을 수행하고 캐시 파일로 저장합니다."""

    processed_path = os.path.join(DATA_DIR, f"{symbol}_processed.csv")
    raw_path       = os.path.join(DATA_DIR, f"{symbol}_5m_raw.csv")
    cache_npz      = os.path.join(DATA_DIR, f"{symbol}_backtest_cache.npz")
    cache_meta     = os.path.join(DATA_DIR, f"{symbol}_backtest_meta.json")

    # ── 파일 존재 확인 ───────────────────────────────────────
    if not os.path.exists(processed_path):
        print(f"[SKIP] {processed_path} 파일이 없습니다.")
        return
    if not os.path.exists(raw_path):
        print(f"[SKIP] {raw_path} 파일이 없습니다.")
        return

    print(f"\n[CACHE] ━━━ {symbol} 캐시 생성 시작 ━━━")

    # ── 1. 데이터 로드 및 병합 ───────────────────────────────
    df     = pd.read_csv(processed_path)
    df_raw = pd.read_csv(raw_path)

    # 학습 코드와 동일하게 ATR 제거
    if 'atr' in df.columns:
        df.drop(columns=['atr'], inplace=True)

    df = pd.merge(
        df,
        df_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']],
        on='timestamp',
        suffixes=('', '_raw'),
    )
    print(f"[CACHE]   로드 완료: {len(df):,}행")

    # ── 2. 1시간봉 추세 피처 (학습 코드 동일) ──────────────────
    df['1h_ema_50']  = df['close_raw'].ewm(span=12 * 50,  adjust=False).mean()
    df['1h_ema_200'] = df['close_raw'].ewm(span=12 * 200, adjust=False).mean()
    df['1h_trend']   = np.where(df['1h_ema_50'] > df['1h_ema_200'], 1, -1)

    # ── 3. 시간 주기 피처 ──────────────────────────────────────
    dt = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)

    # ── 4. 원본 가격·날짜 백업 (실제 수익 계산용) ───────────────
    raw_close    = df['close_raw'].values.copy()
    raw_dates_ms = df['timestamp'].values.copy()   # int64 ms 단위 저장

    # ── 5. 가격/거래량 스케일링 (학습 코드 동일) ────────────────
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = df[f'{col}_raw'].pct_change().fillna(0)
        q_lo, q_hi = df[col].quantile(0.001), df[col].quantile(0.999)
        df[col] = df[col].clip(q_lo, q_hi)

    vol_col = ['volume']
    vol_ma  = df['volume_raw'].rolling(24).mean() + 1e-9
    df[vol_col[0]] = (df['volume_raw'] / vol_ma).clip(0, 10)

    # ── 6. 임시 원본 컬럼 제거 및 결측치 처리 ──────────────────
    drop_cols = [c for c in df.columns if c.endswith('_raw')]
    df.drop(columns=drop_cols, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    valid_mask   = df.notna().all(axis=1).values
    df           = df[valid_mask]
    raw_close    = raw_close[valid_mask]
    raw_dates_ms = raw_dates_ms[valid_mask]

    # ── 7. 컬럼 순서 정렬 (멀티 브랜치: 가격4 → 거래량1 → 보조지표N) ──
    exclude_cols = ['timestamp', 'datetime', 'Target', '1h_ema_50', '1h_ema_200']
    ind_cols     = [c for c in df.columns if c not in price_cols + vol_col + exclude_cols]
    feature_cols = price_cols + vol_col + ind_cols

    features = df[feature_cols].values.astype(np.float32)
    val_end  = int(len(features) * 0.85)

    # ── 8. 캐시 저장 ────────────────────────────────────────
    np.savez_compressed(
        cache_npz,
        features     = features,
        raw_close    = raw_close.astype(np.float64),
        raw_dates_ms = raw_dates_ms.astype(np.int64),
    )

    meta = {
        'symbol'      : symbol,
        'ind_cols'    : ind_cols,
        'feature_cols': feature_cols,
        'val_end'     : val_end,
        'total_rows'  : len(features),
        'num_features': features.shape[1],
        'num_indicators': len(ind_cols),
    }
    with open(cache_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[CACHE]   피처 행렬  : {features.shape}  →  {cache_npz}")
    print(f"[CACHE]   메타데이터 : {cache_meta}")
    print(f"[CACHE]   보조지표 수: {len(ind_cols)}개  /  테스트 시작 인덱스: {val_end}")
    print(f"[CACHE] ✅ {symbol} 캐시 생성 완료\n")


if __name__ == "__main__":
    for sym in SYMBOLS:
        build_cache(sym)
    print("=" * 50)
    print("모든 심볼의 캐시 생성이 완료되었습니다.")
    print("이제 crypto_backtester.py 를 실행하면 캐시를 자동으로 불러옵니다.")
    print("=" * 50)
