import os
import json
import time
import argparse
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
# 앙상블 모델 로더 및 예측 함수
# ─────────────────────────────────────────────────────────────
def _load_ensemble_models(model_dir, num_models, device):
    """앙상블 디렉토리에서 모델들 로드"""
    models = []
    for i in range(num_models):
        model_path = os.path.join(model_dir, f"model_{i}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"앙상블 모델을 찾을 수 없습니다: {model_path}")
        
        # 모델 생성 (num_indicators, num_patterns는 현재 데이터와 일치해야 함)
        model = MultiBranchCryptoPredictor(num_indicators=19, num_patterns=10, dropout=0.3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    
    print(f"[INFO] 앙상블 모델 로드 완료: {num_models}개 모델")
    return models


def _ensemble_predict_soft(models, batch):
    """
    Soft Voting: 모든 모델의 확률을 평균
    
    Returns:
    --------
    ensemble_probs : np.ndarray (batch_size, 3)
        평균 확률
    ensemble_stds : np.ndarray (batch_size, 3)
        각 클래스 확률의 표준편차
    agreement_count : np.ndarray (batch_size,)
        가장 높은 확률 클래스에 동의한 모델 개수
    """
    all_probs = []
    
    with torch.no_grad():
        for model in models:
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    # (num_models, batch_size, 3) → (batch_size, 3)
    all_probs = np.array(all_probs)  # (num_models, batch_size, 3)
    ensemble_probs = all_probs.mean(axis=0)  # 평균
    ensemble_stds = all_probs.std(axis=0)    # 표준편차
    
    # 합의도 계산: 각 샘플에서 주요 클래스로 투표한 모델 개수
    main_class = ensemble_probs.argmax(axis=1)  # (batch_size,)
    agreement_count = np.zeros(len(batch))
    for i in range(len(batch)):
        for j, probs in enumerate(all_probs):
            if probs[i].argmax() == main_class[i]:
                agreement_count[i] += 1
    
    return ensemble_probs, ensemble_stds, agreement_count


def _ensemble_predict_hard(models, batch):
    """
    Hard Voting: 모델들의 다수결
    
    Returns:
    --------
    ensemble_class : np.ndarray (batch_size,)
        투표 결과 클래스
    agreement_count : np.ndarray (batch_size,)
        해당 클래스에 투표한 모델 개수
    """
    all_preds = []
    
    with torch.no_grad():
        for model in models:
            outputs = model(batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu().numpy())
    
    all_preds = np.array(all_preds)  # (num_models, batch_size)
    
    # 다수결
    ensemble_class = np.zeros(all_preds.shape[1], dtype=int)
    agreement_count = np.zeros(all_preds.shape[1], dtype=int)
    
    for i in range(all_preds.shape[1]):
        votes = all_preds[:, i]
        unique, counts = np.unique(votes, return_counts=True)
        ensemble_class[i] = unique[counts.argmax()]
        agreement_count[i] = counts.max()
    
    return ensemble_class, agreement_count


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
                 batch_size=512, tp_pct=0.015, sl_pct=-0.007, max_bars=72):
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

    # 배리어 설정
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


def run_signal_tracker(data_path, model_path, seq_length=120, threshold_prob=0.35,
                       batch_size=512, tp_pct=0.015, sl_pct=-0.005,
                       horizon=72, use_ensemble=False, ensemble_mode='soft', ensemble_count=5):
    """
    실제 매매 없이 모델 신호의 승/패만 추적하는 리포트 모드.
    - 신호 발생 조건: long_prob 또는 short_prob >= threshold_prob
    - 결과 판정: horizon 내 TP/SL 선터치 우선, 미터치 시 horizon 종료 수익률 부호로 판정
    """
    cached = _load_cache(data_path)

    if cached is not None:
        features, raw_close, raw_dates, meta = cached
        num_indicators = meta['num_indicators']
        val_end = meta['val_end']
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

        df['1h_ema_50'] = df['close_raw'].ewm(span=12 * 50, adjust=False).mean()
        df['1h_ema_200'] = df['close_raw'].ewm(span=12 * 200, adjust=False).mean()
        df['1h_trend'] = np.where(df['1h_ema_50'] > df['1h_ema_200'], 1, -1)

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
        vol_ma = df['volume_raw'].rolling(24).mean() + 1e-9
        df[vol_col[0]] = (df['volume_raw'] / vol_ma).clip(0, 10)

        drop_cols = [c for c in df.columns if c.endswith('_raw')]
        df.drop(columns=drop_cols, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        valid_mask = df.notna().all(axis=1).values
        df = df[valid_mask]
        raw_close = raw_close[valid_mask]
        raw_dates = raw_dates[valid_mask]

        exclude_cols = ['timestamp', 'datetime', 'Target', '1h_ema_50', '1h_ema_200']
        ind_cols = [c for c in df.columns if c not in price_cols + vol_col + exclude_cols]
        feature_cols = price_cols + vol_col + ind_cols
        features = df[feature_cols].values.astype(np.float32)

        num_indicators = len(ind_cols)
        val_end = int(len(features) * 0.85)

    test_features = features[val_end:]
    test_close = raw_close[val_end:]
    test_dates = raw_dates[val_end:]

    print(f"[INFO] 신호 추적 기간: {pd.to_datetime(test_dates[0])} ~ {pd.to_datetime(test_dates[-1])}")
    print(f"[INFO] 테스트 샘플 수: {len(test_features):,}개")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiBranchCryptoPredictor(num_indicators=num_indicators, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"[INFO] 배치 추론 시작 (batch_size={batch_size}, device={device})...")
    t_infer = time.time()
    dataset = SlidingWindowDataset(test_features, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    predictions = []
    prediction_stds = []
    agreement_counts = []

    if use_ensemble:
        # 앙상블 모드
        ensemble_models = _load_ensemble_models("models/ensemble", ensemble_count, device)
        print(f"[INFO] 앙상블 추론 시작 (모드: {ensemble_mode}, {ensemble_count}개 모델)...")
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                
                if ensemble_mode == 'soft':
                    # Soft Voting
                    probs, stds, agreement = _ensemble_predict_soft(ensemble_models, batch)
                    predictions.extend(probs)
                    prediction_stds.extend(stds)
                    agreement_counts.extend(agreement)
                else:
                    # Hard Voting → 확률로 변환
                    class_pred, agreement = _ensemble_predict_hard(ensemble_models, batch)
                    # Hard voting 결과를 확률로 표현
                    probs_hard = np.zeros((len(batch), 3))
                    for i, cls in enumerate(class_pred):
                        probs_hard[i, cls] = agreement[i] / ensemble_count
                    predictions.extend(probs_hard)
                    agreement_counts.extend(agreement)
                    prediction_stds.extend(np.zeros((len(batch), 3)))
    else:
        # 단일 모델 모드
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                probs = torch.softmax(model(batch), dim=1).cpu().numpy()
                predictions.extend(probs)
                prediction_stds.extend(np.zeros_like(probs))
                agreement_counts.extend(np.ones(len(batch)) * 1)

    print(f"[INFO] 추론 완료 — {len(predictions):,}개 예측 ({time.time()-t_infer:.1f}초)")
    if use_ensemble:
        print(f"[INFO] 앙상블 모드: {ensemble_mode.upper()} ({ensemble_count}개 모델)")
    print(f"[INFO] 신호 조건: prob >= {threshold_prob:.2f}, TP={tp_pct*100:.2f}%, SL={sl_pct*100:.2f}%, horizon={horizon} bars")

    long_signal_logs = []
    short_signal_logs = []
    long_total = 0
    short_total = 0
    long_win = 0
    short_win = 0

    for i, probs in enumerate(predictions):
        entry_idx = i + seq_length
        if entry_idx >= len(test_close):
            break

        long_prob = float(probs[1])
        short_prob = float(probs[2])

        signal_type = None
        signal_prob = 0.0

        if long_prob >= threshold_prob and long_prob >= short_prob:
            signal_type = 'Long'
            signal_prob = long_prob
        elif short_prob >= threshold_prob and short_prob > long_prob:
            signal_type = 'Short'
            signal_prob = short_prob

        if signal_type is None:
            continue

        entry_price = float(test_close[entry_idx])
        entry_date = pd.to_datetime(test_dates[entry_idx])
        end_idx = min(entry_idx + horizon, len(test_close) - 1)

        outcome = 'LOSS'
        reason = 'SL/음수 종료'
        realized_ret = 0.0
        exit_idx = end_idx
        
        # MFE/MAE 계산용 가격 추적
        max_price = entry_price
        min_price = entry_price

        for j in range(entry_idx + 1, end_idx + 1):
            price_j = float(test_close[j])
            max_price = max(max_price, price_j)
            min_price = min(min_price, price_j)
            
            if signal_type == 'Long':
                ret = (price_j - entry_price) / entry_price
            else:
                ret = (entry_price - price_j) / entry_price

            if ret >= tp_pct:
                outcome = 'WIN'
                reason = 'TP 선터치'
                realized_ret = ret
                exit_idx = j
                break
            if ret <= sl_pct:
                outcome = 'LOSS'
                reason = 'SL 선터치'
                realized_ret = ret
                exit_idx = j
                break
        else:
            final_price = float(test_close[end_idx])
            if signal_type == 'Long':
                realized_ret = (final_price - entry_price) / entry_price
            else:
                realized_ret = (entry_price - final_price) / entry_price
            if realized_ret > 0:
                outcome = 'WIN'
                reason = '시간종료 양수'
        
        # MFE/MAE 계산
        if signal_type == 'Long':
            mfe = (max_price - entry_price) / entry_price
            mae = (min_price - entry_price) / entry_price
        else:
            mfe = (entry_price - min_price) / entry_price
            mae = (entry_price - max_price) / entry_price

        bars_held = exit_idx - entry_idx
        exit_date = pd.to_datetime(test_dates[exit_idx])

        # 앙상블 신뢰도 계산
        class_idx = 1 if signal_type == 'Long' else 2
        ensemble_avg_prob = float(probs[class_idx])
        ensemble_std = float(prediction_stds[i][class_idx])
        agreement = int(agreement_counts[i])
        confidence = (agreement / (ensemble_count if use_ensemble else 1)) * 100

        signal_log = {
            'entry_date': entry_date,
            'type': signal_type,
            'prob': signal_prob,
            'outcome': outcome,
            'ret_pct': realized_ret * 100,
            'bars': bars_held,
            'reason': reason,
            'exit_date': exit_date,
            'mfe_pct': mfe * 100,
            'mae_pct': mae * 100,
            'ensemble_avg_prob': ensemble_avg_prob,
            'ensemble_std': ensemble_std,
            'agreement': agreement,
            'confidence': confidence,
        }
        
        if signal_type == 'Long':
            long_signal_logs.append(signal_log)
        else:
            short_signal_logs.append(signal_log)

        if signal_type == 'Long':
            long_total += 1
            if outcome == 'WIN':
                long_win += 1
        else:
            short_total += 1
            if outcome == 'WIN':
                short_win += 1

    total_signals = len(long_signal_logs) + len(short_signal_logs)
    long_loss = long_total - long_win
    short_loss = short_total - short_win
    total_win = long_win + short_win
    total_loss = total_signals - total_win

    long_wr = (long_win / long_total * 100) if long_total else 0.0
    short_wr = (short_win / short_total * 100) if short_total else 0.0
    total_wr = (total_win / total_signals * 100) if total_signals else 0.0

    print("\n========================================")
    print("📡 신호 추적 리포트 (매매 미실행)")
    print("========================================")
    print(f"총 신호 횟수 : {total_signals:>5}회")
    print(f"전체 적중률  : {total_wr:>6.2f}%  ({total_win}승 / {total_loss}패)")
    print("----------------------------------------")
    print(f"[Long ]  총 {long_total:>4}회  |  적중률 {long_wr:>6.2f}%  |  {long_win}승 / {long_loss}패")
    print(f"[Short]  총 {short_total:>4}회  |  적중률 {short_wr:>6.2f}%  |  {short_win}승 / {short_loss}패")
    print("========================================")

    if total_signals == 0:
        print("[INFO] 조건을 만족하는 신호가 없습니다. threshold_prob를 낮춰보세요.")
        return

    # 신호 로그를 포지션별로 분리해서 저장
    symbol = data_path.split('/')[-1].replace('_processed.csv', '')
    
    def save_signal_log(signal_logs, position_type):
        """신호 로그를 파일로 저장하는 헬퍼 함수"""
        log_filename = f"signal_logs_{symbol}_{position_type.lower()}.txt"
        
        if not signal_logs:
            print(f"[INFO] {position_type} 신호가 없습니다.")
            return
        
        wr = (sum(1 for log in signal_logs if log['outcome'] == 'WIN') / len(signal_logs) * 100) if signal_logs else 0.0
        
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write(f"신호 추적 로그 - {symbol} ({position_type})\n")
            f.write(f"생성 일시: {pd.Timestamp.now()}\n")
            f.write(f"신호 조건: prob >= {threshold_prob:.2f}, TP={tp_pct*100:.2f}%, SL={sl_pct*100:.2f}%, horizon={horizon} bars\n")
            if use_ensemble:
                f.write(f"앙상블: {ensemble_mode.upper()} ({ensemble_count}개 모델)\n")
            f.write(f"\n총 신호 {len(signal_logs)}건 | 적중률 {wr:.2f}%\n")
            f.write("\n" + "="*160 + "\n")
            
            if use_ensemble:
                f.write("date                | prob   | MFE%    | MAE%    | result | ret%    | bars | confidence | reason\n")
                f.write("-"*160 + "\n")
                for row in signal_logs:
                    f.write(
                        f"{row['entry_date']} | "
                        f"{row['prob']*100:>5.2f}% | "
                        f"{row['mfe_pct']:>+7.3f}% | "
                        f"{row['mae_pct']:>+7.3f}% | "
                        f"{row['outcome']:<6} | "
                        f"{row['ret_pct']:>+7.3f}% | "
                        f"{row['bars']:>4} | "
                        f"{row['confidence']:>6.1f}% ({int(row['agreement'])}/{ensemble_count}) | "
                        f"{row['reason']}\n"
                    )
            else:
                f.write("date                | prob   | MFE%    | MAE%    | result | ret%    | bars | reason\n")
                f.write("-"*160 + "\n")
                for row in signal_logs:
                    f.write(
                        f"{row['entry_date']} | "
                        f"{row['prob']*100:>5.2f}% | "
                        f"{row['mfe_pct']:>+7.3f}% | "
                        f"{row['mae_pct']:>+7.3f}% | "
                        f"{row['outcome']:<6} | "
                        f"{row['ret_pct']:>+7.3f}% | "
                        f"{row['bars']:>4} | "
                        f"{row['reason']}\n"
                    )
        
        print(f"[INFO] ✅ {position_type} 신호 로그 저장 완료: {log_filename} ({len(signal_logs)}건)")
    
    save_signal_log(long_signal_logs, 'Long')
    save_signal_log(short_signal_logs, 'Short')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto backtest / signal tracker runner")
    parser.add_argument("--mode", choices=["backtest", "signal", "both"], default="both")
    parser.add_argument("--data-path", default="data/BTC_USDT_processed.csv")
    parser.add_argument("--model-path", default="models/best_lstm_btc_5m_multibranch_v4.pth")
    parser.add_argument("--seq-length", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tp-pct", type=float, default=0.045,
                        help="Take profit ratio. e.g. 0.045 = 4.5%")
    parser.add_argument("--sl-pct", type=float, default=-0.015,
                        help="Stop loss ratio. e.g. -0.015 = -1.5%")
    parser.add_argument("--horizon", type=int, default=72,
                        help="Maximum holding bars / signal evaluation horizon")
    parser.add_argument("--use-ensemble", action="store_true",
                        help="Use ensemble models instead of single model")
    parser.add_argument("--ensemble-mode", choices=["soft", "hard"], default="soft",
                        help="Ensemble voting mode (soft=probability average, hard=majority vote)")
    parser.add_argument("--ensemble-count", type=int, default=5,
                        help="Number of ensemble models to load (3, 4, or 5)")
    args = parser.parse_args()

    if args.mode in ["backtest", "both"]:
        print("\n================ BACKTEST LOG ================")
        run_backtest(
            data_path=args.data_path,
            model_path=args.model_path,
            seq_length=args.seq_length,
            threshold_prob=args.threshold,
            batch_size=args.batch_size,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            max_bars=args.horizon,
        )

    if args.mode in ["signal", "both"]:
        print("\n================ SIGNAL LOG ==================")
        run_signal_tracker(
            data_path=args.data_path,
            model_path=args.model_path,
            seq_length=args.seq_length,
            threshold_prob=args.threshold,
            batch_size=args.batch_size,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            horizon=args.horizon,
            use_ensemble=args.use_ensemble,
            ensemble_mode=args.ensemble_mode,
            ensemble_count=args.ensemble_count,
        )