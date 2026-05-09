import os
import torch
import numpy as np
import pandas as pd
import talib
import argparse
import sys
from torch.utils.data import Dataset, DataLoader
import time
import warnings
import logging

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in os.sys.path:
    os.sys.path.insert(0, ROOT_DIR)

from src.models.base_models import PriceActionExpert, ContextExpert

# ── 로깅 설정 ─────────────────────────────────────────────────────────────
os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "orchestrator.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ExtractSignals")

DEFAULT_THRESHOLD_SETS = [
    ("conservative", 0.68, 0.22, 0.50, 0.03),
    ("balanced", 0.65, 0.18, 0.45, 0.02),
    ("permissive", 0.62, 0.14, 0.40, 0.01),
]

class SlidingWindowDataset(Dataset):
    def __init__(self, features, seq_length):
        self.features = features
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx : idx + self.seq_length], dtype=torch.float32)


def _print_period_summary(results_df):
    tmp = results_df.copy()
    tmp['datetime'] = pd.to_datetime(tmp['datetime'], errors='coerce')
    tmp = tmp.dropna(subset=['datetime'])

    if tmp.empty:
        logger.warning("\n[연/분기 리포트] datetime 파싱 실패로 생성할 수 없습니다.")
        return

    tmp['year'] = tmp['datetime'].dt.year.astype(int)
    tmp['quarter'] = tmp['datetime'].dt.to_period('Q').astype(str)

    year_stats = (
        tmp.groupby('year')[['long_score', 'short_score', 'context_score']]
        .agg(['mean', 'std', 'min', 'max'])
        .round(4)
    )

    quarter_stats = (
        tmp.groupby('quarter')[['long_score', 'short_score', 'context_score']]
        .agg(['mean', 'std'])
        .round(4)
    )

    logger.info("")
    logger.info(f"[연도별 점수 통계]\n{year_stats}")
    logger.info("")
    logger.info(f"[분기별 점수 통계]\n{quarter_stats}")


def _scan_threshold_sets(results_df, threshold_sets):
    rows = []
    n = len(results_df)
    if n == 0:
        return pd.DataFrame()

    for name, long_th, short_th, ctx_th, margin in threshold_sets:
        long_sig = (
            (results_df['long_score'] >= long_th)
            & (results_df['context_score'] >= ctx_th)
            & (results_df['long_score'] >= results_df['short_score'] + margin)
        )
        short_sig = (
            (results_df['short_score'] >= short_th)
            & (results_df['context_score'] >= ctx_th)
            & (results_df['short_score'] >= results_df['long_score'] + margin)
        )

        both = long_sig & short_sig
        long_only = long_sig & ~short_sig
        short_only = short_sig & ~long_sig
        any_sig = long_only | short_only

        long_cnt = int(long_only.sum())
        short_cnt = int(short_only.sum())
        both_cnt = int(both.sum())
        any_cnt = int(any_sig.sum())
        dir_total = long_cnt + short_cnt
        long_bias = (long_cnt / dir_total * 100.0) if dir_total > 0 else 0.0

        rows.append({
            'set': name,
            'long_th': long_th,
            'short_th': short_th,
            'ctx_th': ctx_th,
            'margin': margin,
            'long_only_count': long_cnt,
            'short_only_count': short_cnt,
            'both_count': both_cnt,
            'any_signal_count': any_cnt,
            'any_signal_rate_pct': any_cnt / n * 100.0,
            'long_bias_pct': long_bias,
        })

    return pd.DataFrame(rows).sort_values('set').reset_index(drop=True)


def _parse_threshold_sets(raw):
    if not raw:
        return DEFAULT_THRESHOLD_SETS

    parsed = []
    tokens = [x.strip() for x in str(raw).split(';') if x.strip()]
    for i, token in enumerate(tokens, start=1):
        parts = [p.strip() for p in token.split(',')]
        if len(parts) != 5:
            raise ValueError(
                "--threshold-sets 형식 오류. 예: "
                "balanced,0.65,0.18,0.45,0.02;permissive,0.62,0.14,0.40,0.01"
            )
        name = parts[0]
        long_th, short_th, ctx_th, margin = map(float, parts[1:])
        parsed.append((name or f"set{i}", long_th, short_th, ctx_th, margin))

    return parsed

def extract_base_signals(data_path, seq_length=120, batch_size=512,
                         threshold_sets=None, output_filename="base_signals_log.csv"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"[INFO] 테스트 데이터를 위한 전체 시계열 로딩 중...")

    # 1. 데이터 로드 및 피처 생성 (다운샘플링 X)
    df = pd.read_csv(data_path)
    if 'atr' in df.columns:
        df.drop(columns=['atr'], inplace=True)

    # raw 경로 결정: 명시적 인자 우선, 미지정 시 data_path 기반 자동 생성
    if args.raw_data_path:
        raw_filepath = args.raw_data_path
    else:
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

    logger.info(f"[INFO] 3개의 전문가 모델 로딩 중...")
    model_dir = os.path.join(ROOT_DIR, "checkpoints", "base_experts")

    long_model = PriceActionExpert().to(device)
    long_model.load_state_dict(torch.load(os.path.join(model_dir, "long_expert.pth"), map_location=device))
    long_model.eval()

    short_model = PriceActionExpert().to(device)
    short_model.load_state_dict(torch.load(os.path.join(model_dir, "short_expert.pth"), map_location=device))
    short_model.eval()

    context_model = ContextExpert(input_dim=len(context_cols)).to(device)
    context_model.load_state_dict(torch.load(os.path.join(model_dir, "context_expert.pth"), map_location=device))
    context_model.eval()

    logger.info(f"[INFO] 배치 추론 시작...")
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

    logger.info(f"[INFO] 추론 완료 ({time.time() - t0:.1f}초)")

    # 결과 저장
    results_df = pd.DataFrame({
        'datetime': test_dates[seq_length:seq_length+len(long_scores)],
        'close': test_close[seq_length:seq_length+len(long_scores)],
        'long_score': np.round(long_scores, 4),
        'short_score': np.round(short_scores, 4),
        'context_score': np.round(context_scores, 4)
    })

    output_dir = os.path.join(ROOT_DIR, "data", "signals")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    results_df.to_csv(out_path, index=False)
    logger.info(f"✅ 1, 2차 모델 점수 로그가 저장되었습니다: {out_path}")
    
    # 간략한 분포 리포트
    logger.info("")
    logger.info(f"[점수 분포 요약]\n{results_df[['long_score', 'short_score', 'context_score']].describe()}")

    _print_period_summary(results_df)

    if threshold_sets is None:
        threshold_sets = DEFAULT_THRESHOLD_SETS
    scan_df = _scan_threshold_sets(results_df, threshold_sets)
    if not scan_df.empty:
        logger.info("")
        logger.info(f"[임계치 세트 재검증]\n{scan_df.to_string(index=False)}")

        scan_path = os.path.join(output_dir, "base_signals_threshold_scan.csv")
        scan_df.to_csv(scan_path, index=False)
        logger.info(f"[INFO] 임계치 재검증 결과 저장: {scan_path}")

        by_year_rows = []
        tmp = results_df.copy()
        tmp['datetime'] = pd.to_datetime(tmp['datetime'], errors='coerce')
        tmp = tmp.dropna(subset=['datetime'])
        if not tmp.empty:
            tmp['year'] = tmp['datetime'].dt.year.astype(int)
            for year, g in tmp.groupby('year'):
                y_scan = _scan_threshold_sets(g, threshold_sets)
                if y_scan.empty:
                    continue
                y_scan.insert(0, 'year', int(year))
                by_year_rows.append(y_scan)
            if by_year_rows:
                by_year_df = pd.concat(by_year_rows, ignore_index=True)
                logger.info("")
                logger.info(f"[임계치 세트 재검증 - 연도별]\n{by_year_df.to_string(index=False)}")
                by_year_path = os.path.join(output_dir, "base_signals_threshold_scan_by_year.csv")
                by_year_df.to_csv(by_year_path, index=False)
                logger.info(f"[INFO] 연도별 임계치 재검증 결과 저장: {by_year_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Base 신호 검증 및 통계 리포트")
    parser.add_argument("--data-path", type=str,
                        default=os.path.join(ROOT_DIR, "data", "BTC_USDT_processed.csv"),
                        help="검증 대상 processed CSV")
    parser.add_argument("--raw-data-path", type=str, default=None,
                        help="Raw CSV 경로 (지정 시 우선). 미지정 시 --data-path 기반 자동 생성")
    parser.add_argument("--seq-length", type=int, default=120,
                        help="입력 시퀀스 길이")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="추론 배치 크기")
    parser.add_argument("--output-filename", type=str, default="base_signals_log.csv",
                        help="data/signals 하위 출력 파일명")
    parser.add_argument(
        "--threshold-sets",
        type=str,
        default="",
        help=(
            "세미콜론 구분 임계치 세트. 형식: name,long_th,short_th,ctx_th,margin;... "
            "예: balanced,0.65,0.18,0.45,0.02;permissive,0.62,0.14,0.40,0.01"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.join(ROOT_DIR, "checkpoints", "base_experts", "long_expert.pth")):
        logger.error("[ERROR] Base 모델이 없습니다. 'python train_base_models.py'를 먼저 실행하세요.")
    else:
        extract_base_signals(
            data_path=args.data_path,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            threshold_sets=_parse_threshold_sets(args.threshold_sets),
            output_filename=args.output_filename,
        )