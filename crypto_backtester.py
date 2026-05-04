import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
# 기존에 만든 모델 구조를 불러옵니다. (파일명이 crypto_model_training.py 여야 합니다)
from crypto_model_training import CryptoPredictorLSTM 

import warnings
warnings.filterwarnings('ignore')

def run_backtest(data_path, model_path, seq_length=120, threshold=0.35):
    print(f"[INFO] 백테스트 데이터 준비 중: {data_path}")
    df = pd.read_csv(data_path)

    # 시간 주기 피처 복원
    if 'datetime' in df.columns:
        dt = pd.to_datetime(df['datetime'])
    else:
        dt = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
    df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin']  = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos']  = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # 🌟 원본 종가와 날짜 보존 (실제 모의투자 수익금 계산용)
    raw_close = df['close'].values.copy()
    raw_dates = dt.values

    # 피처 변환 (학습 때와 완벽히 동일한 환경 구성)
    price_volume_cols = ['open', 'high', 'low', 'close', 'volume']
    df_features = df.copy()
    df_features[price_volume_cols] = df_features[price_volume_cols].pct_change().fillna(0)

    for col in price_volume_cols:
        q_lo = df_features[col].quantile(0.001)
        q_hi = df_features[col].quantile(0.999)
        df_features[col] = df_features[col].clip(q_lo, q_hi)

    # 결측치 제거 후 인덱스 정렬
    valid_indices = df_features.dropna().index
    df_features = df_features.loc[valid_indices]
    raw_close = raw_close[valid_indices]
    raw_dates = raw_dates[valid_indices]

    exclude_cols = ['timestamp', 'datetime', 'Target']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    features = df_features[feature_cols].values

    # 학습 때 분리했던 Test 구간(마지막 15%)으로 정확히 이동
    n = len(features)
    val_end = int(n * 0.85)

    test_features = features[val_end:]
    test_close = raw_close[val_end:]
    test_dates = raw_dates[val_end:]

    print(f"[INFO] 테스트 기간: {pd.to_datetime(test_dates[0])} ~ {pd.to_datetime(test_dates[-1])}")

    # 인공지능 두뇌(가중치) 이식
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CryptoPredictorLSTM(input_size=len(feature_cols), hidden_size=256, num_layers=3, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("[INFO] AI가 과거 차트를 보며 5분 단위로 예측을 수행 중입니다...")
    predictions = []
    with torch.no_grad():
        for i in range(len(test_features) - seq_length):
            x = torch.tensor(test_features[i : i + seq_length], dtype=torch.float32).unsqueeze(0).to(device)
            pred = torch.sigmoid(model(x)).item()
            predictions.append(pred)

    # 🚀 백테스트(모의투자) 시뮬레이션 엔진 🚀
    initial_balance = 10000.0
    print(f"\n[INFO] 💸 시뮬레이션 시작 (초기 자본: {initial_balance:,.0f} USDT, 진입 임계값: {threshold})")
    balance = initial_balance
    equity_curve = []
    trades = []

    holding = False
    entry_price = 0.0
    bars_held = 0

    # 트리플 배리어 설정
    tp_pct = 0.02    # 익절 (+2%)
    sl_pct = -0.01   # 손절 (-1%)
    max_bars = 72    # 6시간 타임아웃
    fee_rate = 0.0005 # 바이비트 시장가 수수료 (0.05%)

    for i in range(len(predictions)):
        # prediction[i]는 test_close[i + seq_length] 시점의 예측값입니다.
        current_price = test_close[i + seq_length]
        current_date = test_dates[i + seq_length]
        pred = predictions[i]

        if not holding:
            if pred >= threshold:
                # [매수 진입]
                holding = True
                entry_price = current_price
                balance *= (1 - fee_rate) # 살 때 수수료 차감
                bars_held = 0
        else:
            bars_held += 1
            ret = (current_price - entry_price) / entry_price

            sell_reason = None
            if ret >= tp_pct:
                sell_reason = "익절 (+2%)"
            elif ret <= sl_pct:
                sell_reason = "손절 (-1%)"
            elif bars_held >= max_bars:
                sell_reason = "시간초과 청산"

            if sell_reason:
                # [매도 청산]
                balance = balance * (current_price / entry_price) * (1 - fee_rate) # 팔 때 수수료 차감
                holding = False
                trades.append({
                    'sell_date': current_date,
                    'reason': sell_reason,
                    'return_pct': (balance / equity_curve[-1] - 1) * 100 if equity_curve else 0,
                    'balance': balance
                })

        equity_curve.append(balance)

    # 결과 분석
    win_trades = [t for t in trades if t['return_pct'] > 0]
    loss_trades = [t for t in trades if t['return_pct'] <= 0]
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0
    pnl_pct = ((balance - initial_balance) / initial_balance) * 100

    print(f"\n========================================")
    print(f"📊 백테스트 결과 리포트")
    print(f"========================================")
    print(f"총 거래 횟수: {len(trades)}회")
    print(f"승률(Win Rate): {win_rate:.2f}% ({len(win_trades)}승 / {len(loss_trades)}패)")
    print(f"최종 자산: {balance:,.2f} USDT")
    print(f"총 수익률(PnL): {pnl_pct:.2f}%")
    print(f"========================================")

    # 수익률 차트(Equity Curve) 이미지 저장
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Bot's Account Balance", color='#2E86C1', linewidth=1.5)
    plt.axhline(initial_balance, color='#E74C3C', linestyle='--', label="Initial $10,000")
    plt.title("AI Trading Bot Backtest Equity Curve (Test Period)", fontsize=14, fontweight='bold')
    plt.xlabel("Time (5-minute ticks)", fontsize=12)
    plt.ylabel("Balance (USDT)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("backtest_result.png", dpi=300)
    print("[INFO] 📈 봇의 자산 변화 차트가 'backtest_result.png' 파일로 예쁘게 저장되었습니다!")

if __name__ == "__main__":
    # 필요한 라이브러리가 없다면 터미널에서 `pip install matplotlib` 를 실행해주세요.
    run_backtest(
        data_path="data/BTC_USDT_processed.csv",
        model_path="models/best_lstm_btc_5m.pth",
        seq_length=120,
        threshold=0.35 # 학습 결과에서 제시된 최적 임계값
    )