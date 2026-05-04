import os
import json
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# V4 모델의 클래스 구조를 불러옵니다.
from crypto_model_training import MultiBranchCryptoPredictor

import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# 슬라이딩 윈도우 데이터셋 (배치 추론용)
# ─────────────────────────────────────────────────────────────
class SlidingWindowDataset(Dataset):
    """캐시에서 로드된 피처 행렬을 seq_length 단위로 잘라 배치 추론에 활용합니다."""
    def __init__(self, features: np.ndarray, seq_length: int):
        self.features   = features
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        return torch.tensor(
            self.features[idx : idx + self.seq_length],
            dtype=torch.float32,
        )


# ─────────────────────────────────────────────────────────────
# 캐시 로더
# ─────────────────────────────────────────────────────────────
def _cache_paths(data_path: str):
    """data_path로부터 캐시 파일 경로 2개를 반환합니다."""
    base = data_path.replace("_processed.csv", "")
    return base + "_backtest_cache.npz", base + "_backtest_meta.json"


def _load_cache(data_path: str):
    """
    캐시 파일이 존재하면 로드하고 (features, raw_close, raw_dates, meta) 를 반환합니다.
    캐시가 없으면 None 을 반환합니다.
    """
    npz_path, meta_path = _cache_paths(data_path)
    if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
        return None

    t0 = time.time()
    cache = np.load(npz_path)
    features     = cache['features']          # (N, num_features) float32
    raw_close    = cache['raw_close']         # (N,) float64
    raw_dates_ms = cache['raw_dates_ms']      # (N,) int64  ms timestamp

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # ms → numpy datetime64 변환
    raw_dates = pd.to_datetime(raw_dates_ms, unit='ms', utc=True).values

    print(f"[CACHE] ✅ 캐시 로드 완료 ({time.time()-t0:.1f}초)")
    print(f"[CACHE]    피처 행렬: {features.shape}  /  보조지표: {meta['num_indicators']}개")
    return features, raw_close, raw_dates, meta


def run_backtest(data_path, model_path, seq_length=120, threshold_prob=0.50,
                 batch_size=512):
    """
    Parameters
    ----------
    data_path       : processed CSV 경로 (캐시 경로는 자동 유도)
    model_path      : 학습된 .pth 모델 경로
    seq_length      : 입력 시퀀스 길이 (학습 때와 동일하게)
    threshold_prob  : 롱/숏 진입 최소 확률 임계값
    batch_size      : 배치 추론 크기 (클수록 빠름, MPS 메모리 한도 고려)
    """

    # ── 1. 데이터 로드 (캐시 우선, 없으면 전체 파이프라인 실행) ──────
    cached = _load_cache(data_path)

    if cached is not None:
        features, raw_close, raw_dates, meta = cached
        num_indicators = meta['num_indicators']
        val_end        = meta['val_end']
    else:
        print(f"[INFO] 캐시 없음 → 전체 데이터 파이프라인 실행 중 (느림)")
        print(f"[INFO] 다음엔 먼저 'python prepare_backtest_cache.py' 를 실행하세요.")

        df = pd.read_csv(data_path)
        if 'atr' in df.columns:
            df.drop(columns=['atr'], inplace=True)

        raw_filepath = data_path.replace("_processed.csv", "_5m_raw.csv")
        df_raw = pd.read_csv(raw_filepath)
        df = pd.merge(df, df_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']],
                      on='timestamp', suffixes=('', '_raw'))

        df['1h_ema_50']  = df['close_raw'].ewm(span=12 * 50,  adjust=False).mean()
        df['1h_ema_200'] = df['close_raw'].ewm(span=12 * 200, adjust=False).mean()
        df['1h_trend']   = np.where(df['1h_ema_50'] > df['1h_ema_200'], 1, -1)

        dt = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)

        raw_close = df['close_raw'].values.copy()
        raw_dates = dt.values.copy()

        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = df[f'{col}_raw'].pct_change().fillna(0)
            q_lo, q_hi = df[col].quantile(0.001), df[col].quantile(0.999)
            df[col] = df[col].clip(q_lo, q_hi)

        vol_col = ['volume']
        vol_ma  = df['volume_raw'].rolling(24).mean() + 1e-9
        df[vol_col[0]] = (df['volume_raw'] / vol_ma).clip(0, 10)

        drop_cols = [c for c in df.columns if c.endswith('_raw')]
        df.drop(columns=drop_cols, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        valid_mask = df.notna().all(axis=1).values
        df        = df[valid_mask]
        raw_close = raw_close[valid_mask]
        raw_dates = raw_dates[valid_mask]

        exclude_cols = ['timestamp', 'datetime', 'Target', '1h_ema_50', '1h_ema_200']
        ind_cols     = [c for c in df.columns if c not in price_cols + vol_col + exclude_cols]
        feature_cols = price_cols + vol_col + ind_cols
        features     = df[feature_cols].values.astype(np.float32)

        num_indicators = len(ind_cols)
        val_end        = int(len(features) * 0.85)

    # ── 2. Test 구간 슬라이싱 ───────────────────────────────────
    test_features = features[val_end:]
    test_close    = raw_close[val_end:]
    test_dates    = raw_dates[val_end:]

    print(f"[INFO] 테스트 기간: {pd.to_datetime(test_dates[0])} ~ {pd.to_datetime(test_dates[-1])}")
    print(f"[INFO] 테스트 샘플 수: {len(test_features):,}개")

    # ── 3. 모델 로드 ────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model  = MultiBranchCryptoPredictor(num_indicators=num_indicators, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ── 4. 배치 추론 (for 루프 → DataLoader 배치) ───────────────
    print(f"[INFO] 배치 추론 시작 (batch_size={batch_size}, device={device})...")
    t_infer = time.time()

    dataset    = SlidingWindowDataset(test_features, seq_length)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    predictions = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
            predictions.extend(probs)

    print(f"[INFO] 추론 완료 — {len(predictions):,}개 예측 ({time.time()-t_infer:.1f}초)")

    # 3. 🚀 백테스트(모의투자) 시뮬레이션 엔진 🚀
    initial_balance = 10000.0
    print(f"\n[INFO] 💸 시뮬레이션 시작 (초기 자본: {initial_balance:,.0f} USDT)")
    print(f"[INFO] 🛡️ 안전장치 가동: AI의 확신이 {threshold_prob*100:.0f}% 이상일 때만 진입합니다.")
    
    balance = initial_balance
    equity_curve = []
    trades = []

    # 포지션 상태 (0: 무포지션, 1: 롱, -1: 숏)
    current_position = 0 
    entry_price = 0.0
    bars_held = 0

    # 배리어 설정 (단순화를 위해 백테스트에서는 고정폭 사용)
    tp_pct = 0.015    # 익절 (+1.5%)
    sl_pct = -0.007   # 손절 (-0.7%)
    max_bars = 72     # 타임아웃 6시간
    fee_rate = 0.0005 # 시장가 수수료 0.05%

    for i in range(len(predictions)):
        current_price = test_close[i + seq_length]
        current_date = test_dates[i + seq_length]
        
        # probs = [관망 확률, 롱 확률, 숏 확률]
        probs = predictions[i]
        
        if current_position == 0:
            # 🌟 [안전장치] AI가 도박을 거절하고 진짜 확신할 때만 진입
            if probs[1] >= threshold_prob:
                # 롱 진입
                current_position = 1
                entry_price = current_price
                balance *= (1 - fee_rate) 
                bars_held = 0
            elif probs[2] >= threshold_prob:
                # 숏 진입
                current_position = -1
                entry_price = current_price
                balance *= (1 - fee_rate)
                bars_held = 0
        else:
            bars_held += 1
            # 롱/숏에 따른 현재 수익률 계산
            if current_position == 1:
                ret = (current_price - entry_price) / entry_price
            else: # 숏일 경우 가격이 내려가야 수익
                ret = (entry_price - current_price) / entry_price

            sell_reason = None
            if ret >= tp_pct:
                sell_reason = f"{'Long' if current_position == 1 else 'Short'} 익절"
            elif ret <= sl_pct:
                sell_reason = f"{'Long' if current_position == 1 else 'Short'} 손절"
            elif bars_held >= max_bars:
                sell_reason = "시간초과 청산"

            if sell_reason:
                # 청산 (롱이든 숏이든 내 원금에 수익/손실 퍼센트만큼 더해줌)
                balance = balance * (1 + ret) * (1 - fee_rate)
                
                trades.append({
                    'sell_date': current_date,
                    'type': 'Long' if current_position == 1 else 'Short',
                    'reason': sell_reason,
                    'return_pct': ret * 100,
                    'balance': balance
                })
                current_position = 0

        equity_curve.append(balance)

    # 4. 결과 분석
    win_trades  = [t for t in trades if t['return_pct'] > 0]
    loss_trades = [t for t in trades if t['return_pct'] <= 0]
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0
    pnl_pct  = ((balance - initial_balance) / initial_balance) * 100

    # 포지션별 분리
    long_trades  = [t for t in trades if t['type'] == 'Long']
    short_trades = [t for t in trades if t['type'] == 'Short']

    long_win   = [t for t in long_trades  if t['return_pct'] > 0]
    long_loss  = [t for t in long_trades  if t['return_pct'] <= 0]
    short_win  = [t for t in short_trades if t['return_pct'] > 0]
    short_loss = [t for t in short_trades if t['return_pct'] <= 0]

    long_wr  = (len(long_win)  / len(long_trades)  * 100) if long_trades  else 0.0
    short_wr = (len(short_win) / len(short_trades) * 100) if short_trades else 0.0

    print(f"\n========================================")
    print(f"📊 백테스트 결과 리포트 (V4 양방향)")
    print(f"========================================")
    print(f"총 거래 횟수 : {len(trades):>5}회")
    print(f"전체 승률    : {win_rate:>6.2f}%  ({len(win_trades)}승 / {len(loss_trades)}패)")
    print(f"----------------------------------------")
    print(f"[Long ]  총 {len(long_trades):>4}회  |  "
          f"승률 {long_wr:>6.2f}%  |  {len(long_win)}승 / {len(long_loss)}패")
    print(f"[Short]  총 {len(short_trades):>4}회  |  "
          f"승률 {short_wr:>6.2f}%  |  {len(short_win)}승 / {len(short_loss)}패")
    print(f"----------------------------------------")
    print(f"최종 자산    : {balance:>12,.2f} USDT")
    print(f"총 수익률    : {pnl_pct:>+8.2f}%")
    print(f"========================================")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="V4 AI Bot Balance", color='#2ECC71', linewidth=1.5)
    plt.axhline(initial_balance, color='#E74C3C', linestyle='--', label="Initial $10,000")
    plt.title(f"V4 AI Trading Bot Equity Curve (Cutoff: {threshold_prob*100}%)", fontsize=14, fontweight='bold')
    plt.xlabel("Time (5-minute ticks)", fontsize=12)
    plt.ylabel("Balance (USDT)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("backtest_v4_result.png", dpi=300)
    print("[INFO] 📈 결과 차트가 'backtest_v4_result.png' 파일로 저장되었습니다!")

if __name__ == "__main__":
    run_backtest(
        data_path="data/BTC_USDT_processed.csv",
        model_path="models/best_lstm_btc_5m_multibranch_v4.pth",
        seq_length=120,
        # 🌟 안전장치: AI의 롱/숏 확신이 35%를 넘을 때만 진입 (기본값)
        threshold_prob=0.35 
    )