import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# 🌟 V4 모델의 클래스 구조를 불러옵니다.
from crypto_model_training import MultiBranchCryptoPredictor

import warnings
warnings.filterwarnings('ignore')

def run_backtest(data_path, model_path, seq_length=120, threshold_prob=0.50):
    print(f"[INFO] V4 양방향 백테스트 데이터 준비 중: {data_path}")
    
    # 1. 데이터 파이프라인 재현 (학습 때와 완벽히 동일해야 함)
    df = pd.read_csv(data_path)
    if 'atr' in df.columns:
        df.drop(columns=['atr'], inplace=True)

    raw_filepath = data_path.replace("_processed.csv", "_5m_raw.csv")
    df_raw = pd.read_csv(raw_filepath)
    df = pd.merge(df, df_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']], on='timestamp', suffixes=('', '_raw'))

    df['1h_ema_50'] = df['close_raw'].ewm(span=12 * 50, adjust=False).mean()
    df['1h_ema_200'] = df['close_raw'].ewm(span=12 * 200, adjust=False).mean()
    df['1h_trend'] = np.where(df['1h_ema_50'] > df['1h_ema_200'], 1, -1)

    dt = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)

    # 🌟 실제 모의투자를 위한 원본 종가 보존
    raw_close = df['close_raw'].values.copy()
    raw_dates = dt.values.copy()

    # 피처 스케일링
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = df[f'{col}_raw'].pct_change().fillna(0)
        q_lo, q_hi = df[col].quantile(0.001), df[col].quantile(0.999)
        df[col] = df[col].clip(q_lo, q_hi)
    
    vol_ma = df['volume_raw'].rolling(24).mean() + 1e-9
    vol_col = ['volume']
    df[vol_col[0]] = (df['volume_raw'] / vol_ma).clip(0, 10) 

    drop_cols = [c for c in df.columns if c.endswith('_raw')]
    df.drop(columns=drop_cols, inplace=True)
    
    # 결측치 제거 시 인덱스를 추적하여 raw_close와 맞춤
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_indices = df.dropna().index
    df = df.loc[valid_indices]
    raw_close = raw_close[valid_indices]
    raw_dates = raw_dates[valid_indices]

    # 멀티 브랜치 강제 정렬 (가격4 -> 거래량1 -> 보조지표N)
    exclude_cols = ['timestamp', 'datetime', 'Target', '1h_ema_50', '1h_ema_200']
    ind_cols = [col for col in df.columns if col not in price_cols + vol_col + exclude_cols]
    feature_cols = price_cols + vol_col + ind_cols
    features = df[feature_cols].values

    # Test 구간(마지막 15%)으로 이동
    val_end = int(len(features) * 0.85)
    test_features = features[val_end:]
    test_close = raw_close[val_end:]
    test_dates = raw_dates[val_end:]

    print(f"[INFO] 테스트 기간: {pd.to_datetime(test_dates[0])} ~ {pd.to_datetime(test_dates[-1])}")

    # 2. 인공지능 두뇌(V4 가중치) 이식
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiBranchCryptoPredictor(num_indicators=len(ind_cols), dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("[INFO] AI가 과거 차트를 보며 5분 단위로 예측을 수행 중입니다...")
    predictions = []
    with torch.no_grad():
        for i in range(len(test_features) - seq_length):
            x = torch.tensor(test_features[i : i + seq_length], dtype=torch.float32).unsqueeze(0).to(device)
            # Softmax를 씌워 각 클래스의 '확률(%)'을 구합니다.
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
            predictions.append(probs)

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
    win_trades = [t for t in trades if t['return_pct'] > 0]
    loss_trades = [t for t in trades if t['return_pct'] <= 0]
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0
    pnl_pct = ((balance - initial_balance) / initial_balance) * 100

    print(f"\n========================================")
    print(f"📊 백테스트 결과 리포트 (V4 양방향)")
    print(f"========================================")
    print(f"총 거래 횟수: {len(trades)}회")
    print(f"승률(Win Rate): {win_rate:.2f}% ({len(win_trades)}승 / {len(loss_trades)}패)")
    print(f"최종 자산: {balance:,.2f} USDT")
    print(f"총 수익률(PnL): {pnl_pct:.2f}%")
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
        # 🌟 안전장치: AI의 롱/숏 확신이 50%를 넘을 때만 진입 (기본값)
        threshold_prob=0.50 
    )