import os
import torch
import numpy as np
import pandas as pd
import talib
from torch.utils.data import Dataset, DataLoader
from crypto_base_models import PriceActionExpert, ContextExpert
import time
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

class SlidingWindowDataset(Dataset):
    def __init__(self, features, seq_length):
        self.features = features
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx : idx + self.seq_length], dtype=torch.float32)

def extract_base_signals(data_path, seq_length=120, batch_size=512):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] 테스트 데이터를 위한 전체 시계열 로딩 중...")

    # 1. 데이터 로드 및 피처 생성 (다운샘플링 X)
    df = pd.read_csv(data_path)
    if 'atr' in df.columns:
        df.drop(columns=['atr'], inplace=True)

    raw_filepath = data_path.replace("_processed.csv", "_5m_raw.csv")
    df_raw = pd.read_csv(raw_filepath)
    df = pd.merge(df, df_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']], on='timestamp', suffixes=('', '_raw'))

    # 스케일링
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = df[f'{col}_raw'].pct_change().fillna(0)
        q_lo, q_hi = df[col].quantile(0.001), df[col].quantile(0.999)
        df[col] = df[col].clip(q_lo, q_hi)
    
    vol_col = ['volume']
    vol_ma = df['volume_raw'].rolling(24).mean() + 1e-9
    df[vol_col[0]] = (df['volume_raw'] / vol_ma).clip(0, 10) 

    # 캔들 패턴 추출
    o, h, l, c = df['open_raw'].values, df['high_raw'].values, df['low_raw'].values, df['close_raw'].values
    pat_dict = {
        'pat_doji': talib.CDLDOJI(o, h, l, c) / 100.0,
        'pat_hammer': talib.CDLHAMMER(o, h, l, c) / 100.0,
        'pat_engulfing': talib.CDLENGULFING(o, h, l, c) / 100.0,
        'pat_morningstar': talib.CDLMORNINGSTAR(o, h, l, c) / 100.0,
        'pat_eveningstar': talib.CDLEVENINGSTAR(o, h, l, c) / 100.0
    }
    for k, v in pat_dict.items():
        df[k] = v

    raw_close = df['close_raw'].values.copy()
    raw_dates = pd.to_datetime(df['timestamp'], unit='ms', utc=True).values.copy()

    drop_cols = [c for c in df.columns if c.endswith('_raw')]
    df.drop(columns=drop_cols, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    valid_mask = df.notna().all(axis=1).values
    df = df[valid_mask]
    raw_close = raw_close[valid_mask]
    raw_dates = raw_dates[valid_mask]

    price_vol_cols = price_cols + vol_col
    exclude_cols = ['timestamp', 'datetime', '1h_ema_50', '1h_ema_200']
    if 'Target' in df.columns: exclude_cols.append('Target')
    context_cols = [c for c in df.columns if c not in price_vol_cols + exclude_cols]

    # 전체 구간을 대상으로 시그널을 생성합니다.
    # 마지막 15%만 사용하려면 아래 예시처럼 val_end를 조정합니다.
    # val_end = int(len(df) * 0.85)
    val_end = 0
    
    test_pv_features = df[price_vol_cols].values.astype(np.float32)[val_end:]
    test_ctx_features = df[context_cols].values.astype(np.float32)[val_end:]
    test_close = raw_close[val_end:]
    test_dates = raw_dates[val_end:]

    print(f"[INFO] 3개의 전문가 모델 로딩 중...")
    model_dir = os.path.join(ROOT_DIR, "models", "commander", "base")

    long_model = PriceActionExpert().to(device)
    long_model.load_state_dict(torch.load(os.path.join(model_dir, "long_expert.pth"), map_location=device))
    long_model.eval()

    short_model = PriceActionExpert().to(device)
    short_model.load_state_dict(torch.load(os.path.join(model_dir, "short_expert.pth"), map_location=device))
    short_model.eval()

    context_model = ContextExpert(input_dim=len(context_cols)).to(device)
    context_model.load_state_dict(torch.load(os.path.join(model_dir, "context_expert.pth"), map_location=device))
    context_model.eval()

    print(f"[INFO] 배치 추론 시작...")
    pv_dataset = SlidingWindowDataset(test_pv_features, seq_length)
    ctx_dataset = SlidingWindowDataset(test_ctx_features, seq_length)
    
    pv_loader = DataLoader(pv_dataset, batch_size=batch_size, shuffle=False)
    ctx_loader = DataLoader(ctx_dataset, batch_size=batch_size, shuffle=False)

    long_scores, short_scores, context_scores = [], [], []

    t0 = time.time()
    with torch.no_grad():
        for pv_batch, ctx_batch in zip(pv_loader, ctx_loader):
            pv_batch, ctx_batch = pv_batch.to(device), ctx_batch.to(device)
            
            long_scores.extend(long_model(pv_batch).cpu().numpy())
            short_scores.extend(short_model(pv_batch).cpu().numpy())
            context_scores.extend(context_model(ctx_batch).cpu().numpy())

    print(f"[INFO] 추론 완료 ({time.time() - t0:.1f}초)")

    # 결과 저장
    results_df = pd.DataFrame({
        'datetime': test_dates[seq_length:seq_length+len(long_scores)],
        'close': test_close[seq_length:seq_length+len(long_scores)],
        'long_score': np.round(long_scores, 4),
        'short_score': np.round(short_scores, 4),
        'context_score': np.round(context_scores, 4)
    })

    output_dir = os.path.join(ROOT_DIR, "data", "commander")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "base_signals_log.csv")
    results_df.to_csv(out_path, index=False)
    print(f"✅ 1, 2차 모델 점수 로그가 저장되었습니다: {out_path}")
    
    # 간략한 분포 리포트
    print("\n[점수 분포 요약]")
    print(results_df[['long_score', 'short_score', 'context_score']].describe())

if __name__ == "__main__":
    if not os.path.exists(os.path.join(ROOT_DIR, "models", "commander", "base", "long_expert.pth")):
        print("[ERROR] Base 모델이 없습니다. 'python train_base_models.py'를 먼저 실행하세요.")
    else:
        extract_base_signals(os.path.join(ROOT_DIR, "data", "BTC_USDT_processed.csv"))