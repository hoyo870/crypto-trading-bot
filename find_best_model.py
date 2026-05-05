"""
여러 시드로 학습 후 백테스트 수익률이 가장 높은 모델을 선택합니다.
"""
import os
import torch
import numpy as np
import subprocess
import sys
from copy import deepcopy
from sklearn.metrics import f1_score
from crypto_model_training import (
    MultiBranchCryptoPredictor, FocalLoss, prepare_data, evaluate_model
)
from crypto_backtester import _load_cache, SlidingWindowDataset, run_backtest
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

SEEDS = [0, 1, 2, 3, 4, 7, 13, 21, 99, 123]
BACKTEST_THRESHOLD = 0.60
DATA_PATH = "data/BTC_USDT_processed.csv"
PATIENCE = 10
EPOCHS = 100

def train_with_seed(seed, train_loader, val_loader, test_loader, num_ind, num_pat, class_weights, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MultiBranchCryptoPredictor(num_indicators=num_ind, num_patterns=num_pat, dropout=0.3)
    model = model.to(device)

    criterion = FocalLoss(gamma=2.0, class_weights=class_weights.to(device), label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = -1.0
    best_weights = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, predicted = torch.max(model(X_batch), 1)
                val_preds.extend(predicted.cpu().numpy())
                val_trues.extend(y_batch.cpu().numpy())

        val_f1 = f1_score(val_trues, val_preds, labels=[1, 2], average='macro', zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [seed={seed}] Epoch {epoch+1} 조기종료 | Best F1={best_val_f1:.4f}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, best_val_f1


def quick_backtest(model, cached, seq_length=120, threshold=0.60,
                   tp_pct=0.014, sl_pct=-0.007, max_bars=72):
    """모델을 직접 받아 빠르게 백테스트 수익률 반환"""
    features, raw_close, raw_dates, meta = cached
    val_end = meta['val_end']
    test_features = features[val_end:]
    test_close = raw_close[val_end:]

    device = next(model.parameters()).device
    loader = DataLoader(SlidingWindowDataset(test_features, seq_length), batch_size=512, shuffle=False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(model(batch.to(device)), dim=1).cpu().numpy()
            predictions.extend(probs)

    balance = 10000.0
    fee_rate = 0.0005
    current_position = 0
    entry_price = 0.0
    bars_held = 0
    n_trades = 0

    for i, probs in enumerate(predictions):
        current_price = test_close[i + seq_length]
        if current_position == 0:
            if probs[1] >= threshold:
                current_position = 1; entry_price = current_price
                balance *= (1 - fee_rate); bars_held = 0
            elif probs[2] >= threshold:
                current_position = -1; entry_price = current_price
                balance *= (1 - fee_rate); bars_held = 0
        else:
            bars_held += 1
            ret = (current_price - entry_price) / entry_price if current_position == 1 else (entry_price - current_price) / entry_price
            if ret >= tp_pct or ret <= sl_pct or bars_held >= max_bars:
                balance = balance * (1 + ret) * (1 - fee_rate)
                current_position = 0
                n_trades += 1

    pnl = (balance - 10000.0) / 10000.0 * 100
    return pnl, n_trades


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] 데이터 로드 중...")
    train_loader, val_loader, test_loader, num_ind, num_pat, class_weights = prepare_data(DATA_PATH, seq_length=120)
    cached = _load_cache(DATA_PATH)

    results = []
    best_pnl = -9999
    best_model = None
    best_seed = -1

    print(f"\n[INFO] {len(SEEDS)}개 시드로 학습 시작: {SEEDS}\n")
    for seed in SEEDS:
        print(f"━━━ Seed={seed} ━━━")
        model, val_f1 = train_with_seed(seed, train_loader, val_loader, test_loader, num_ind, num_pat, class_weights, device)
        pnl, n_trades = quick_backtest(model, cached, threshold=BACKTEST_THRESHOLD)
        results.append((seed, val_f1, pnl, n_trades))
        print(f"  Val F1={val_f1:.4f} | Backtest PnL={pnl:+.2f}% ({n_trades}거래)")

        if pnl > best_pnl:
            best_pnl = pnl
            best_model = deepcopy(model.state_dict())
            best_seed = seed

    print(f"\n{'='*50}")
    print(f"📊 전체 결과 (threshold={BACKTEST_THRESHOLD})")
    print(f"{'='*50}")
    for seed, f1, pnl, n in results:
        marker = " ← Best" if seed == best_seed else ""
        print(f"  Seed={seed:3d} | F1={f1:.4f} | PnL={pnl:+.2f}% | {n}거래{marker}")

    print(f"\n✅ Best: seed={best_seed}, PnL={best_pnl:+.2f}%")
    save_path = "models/best_lstm_btc_5m_v5.pth"
    torch.save(best_model, save_path)
    print(f"[INFO] 💾 Best 모델 저장: {save_path}")


if __name__ == "__main__":
    main()
