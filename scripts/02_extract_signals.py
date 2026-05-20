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

from src.models.base_models import PriceActionExpert, ContextExpert, _PHASE_BULL_END, _PHASE_VAL_END
from src.utils.platform_utils import get_device, configure_torch, get_optimal_workers, get_pin_memory, log_platform_info

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
    # (name, z_th, ctx_th, z_lead)
    # z_score = (score - global_mean) / global_std  (전체 추출 데이터 기준)
    # long  발화: L_z >= z_th AND (L_z - S_z) >= z_lead AND ctx >= ctx_th
    # short 발화: S_z >= z_th AND (S_z - L_z) >= z_lead AND ctx >= ctx_th
    # z_lead=0  → 정확히 50/50 방향 분리
    # z_lead>0  → 명확한 우위 요구 (fewer signals)
    # z_lead<0  → 격차 허용 (both 포함)
    ("conservative_opt", 0.5, 0.60,  0.3),  # 상위 30%, 명확한 우위
    ("balanced_opt",     0.0, 0.49,  0.0),   # 평균 이상, 상대적 우위만
    ("permissive_opt",  -0.3, 0.45, -0.3),  # 완화, 겹침 허용
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

    # z-score 정규화: 두 모델의 스케일 차이(시스템적 S>L 오프셋) 제거
    L_mean = results_df['long_score'].mean()
    L_std  = results_df['long_score'].std() + 1e-8
    S_mean = results_df['short_score'].mean()
    S_std  = results_df['short_score'].std() + 1e-8
    L_z = (results_df['long_score']  - L_mean) / L_std
    S_z = (results_df['short_score'] - S_mean) / S_std

    for name, z_th, ctx_th, z_lead in threshold_sets:
        # [z-score 기반 대칭 방향 조건]
        # long  발화: L_z >= z_th AND (L_z - S_z) >= z_lead AND ctx >= ctx_th
        # short 발화: S_z >= z_th AND (S_z - L_z) >= z_lead AND ctx >= ctx_th
        long_sig = (
            (L_z >= z_th)
            & ((L_z - S_z) >= z_lead)
            & (results_df['context_score'] >= ctx_th)
        )
        short_sig = (
            (S_z >= z_th)
            & ((S_z - L_z) >= z_lead)
            & (results_df['context_score'] >= ctx_th)
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
            'z_th': z_th,
            'ctx_th': ctx_th,
            'z_lead': z_lead,
            'L_mean': round(L_mean, 4),
            'L_std': round(L_std, 4),
            'S_mean': round(S_mean, 4),
            'S_std': round(S_std, 4),
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
        if len(parts) != 4:
            raise ValueError(
                "--threshold-sets 형식 오류. 예: "
                "balanced,0.0,0.49,0.0;conservative,0.5,0.60,0.3"
                " (name,z_th,ctx_th,z_lead)"
            )
        name = parts[0]
        z_th, ctx_th, z_lead = map(float, parts[1:])
        parsed.append((name or f"set{i}", z_th, ctx_th, z_lead))

    return parsed

def extract_base_signals(data_path, seq_length=120, batch_size=512,
                         threshold_sets=None, output_filename="base_signals_log.csv",
                         raw_data_path=None):
    _dev_str = get_device()
    configure_torch(_dev_str)
    device = torch.device(_dev_str)
    log_platform_info(logger)
    logger.info(f"[INFO] 테스트 데이터를 위한 전체 시계열 로딩 중...")

    # 1. 데이터 로드 및 피처 생성 (다운샘플링 X)
    df = pd.read_csv(data_path)
    if 'atr' in df.columns:
        df.drop(columns=['atr'], inplace=True)

    # raw 경로 결정: 명시적 인자 우선, 미지정 시 data_path 기반 자동 생성
    if raw_data_path:
        raw_filepath = raw_data_path
    else:
        raw_filepath = data_path.replace(
            os.path.join("data", "processed"),
            os.path.join("data", "raw")
        ).replace("_processed.csv", "_5m_raw.csv")
    df_raw = pd.read_csv(raw_filepath)
    df = pd.merge(df, df_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']], on='timestamp', suffixes=('', '_raw'))

    # 스케일링
    # quantile 경계: Phase 0+1 (train ~ 2025-06-30)만으로 산출해 look-ahead 방지.
    price_cols = ['open', 'high', 'low', 'close']
    _dt_q = pd.to_datetime(df['datetime'])
    _q_cutoff = int((_dt_q <= _PHASE_BULL_END).sum())
    if _q_cutoff == 0:
        _q_cutoff = int(len(df) * 0.70)
    # [Sync Fix 2] base_models.py와 동일: prev_close 기준 Log Return (캔들 형태 보존)
    prev_close = df['close_raw'].shift(1)
    for col in price_cols:
        df[col] = np.log(df[f'{col}_raw'] / prev_close).fillna(0)
        q_lo = df[col].iloc[:_q_cutoff].quantile(0.001)
        q_hi = df[col].iloc[:_q_cutoff].quantile(0.999)
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

    # ── [Sync Fix 7] base_models.py 와 동일한 상대 지표 피처 생성 (_raw 제거 전) ──
    # 훈련 시 context 피처 목록과 일치시키기 위해 반드시 여기서 계산합니다.
    _cr_inf      = df['close_raw'].values.astype(np.float64)
    _safe_cr_inf = np.where(_cr_inf > 0, _cr_inf, 1.0)
    _e20  = talib.EMA(_cr_inf, timeperiod=20)
    _e50  = talib.EMA(_cr_inf, timeperiod=50)
    _e200 = talib.EMA(_cr_inf, timeperiod=200)
    df['ema_20_r']  = np.clip(np.where(np.isnan(_e20),  0.0, _e20  / _safe_cr_inf - 1.0), -0.30, 0.30)
    df['ema_50_r']  = np.clip(np.where(np.isnan(_e50),  0.0, _e50  / _safe_cr_inf - 1.0), -0.50, 0.50)
    df['ema_200_r'] = np.clip(np.where(np.isnan(_e200), 0.0, _e200 / _safe_cr_inf - 1.0), -1.00, 1.00)
    _mv, _msv, _mhv = talib.MACD(_cr_inf, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_r']        = np.clip(np.where(np.isnan(_mv),  0.0, _mv  / _safe_cr_inf), -0.05, 0.05)
    df['macd_signal_r'] = np.clip(np.where(np.isnan(_msv), 0.0, _msv / _safe_cr_inf), -0.05, 0.05)
    df['macd_hist_r']   = np.clip(np.where(np.isnan(_mhv), 0.0, _mhv / _safe_cr_inf), -0.03, 0.03)

    drop_cols = [c for c in df.columns if c.endswith('_raw')]
    df.drop(columns=drop_cols, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    valid_mask = df.notna().all(axis=1).values
    df = df[valid_mask]
    raw_close = raw_close[valid_mask]
    raw_dates = raw_dates[valid_mask]

    price_vol_cols = price_cols + vol_col
    # [Sync Fix 7] base_models.py 의 exclude_cols 와 동일하게 유지해야
    # 훈련 시 ContextExpert input_dim 과 추론 시 context_cols 길이가 일치합니다.
    _abs_price_feat = [
        'ema_20', 'ema_50', 'ema_200',
        'bb_upper', 'bb_middle', 'bb_lower',
        'macd', 'macd_signal', 'macd_hist',
    ]
    exclude_cols = (
        ['timestamp', 'datetime', '1h_ema_50', '1h_ema_200', 'market_phase']
        + _abs_price_feat
    )
    if 'Target' in df.columns: exclude_cols.append('Target')
    context_cols = [c for c in df.columns if c not in price_vol_cols + exclude_cols]

    # 전체 구간을 대상으로 시그널을 생성합니다.
    # 마지막 15%만 사용하려면 아래 예시처럼 val_end를 조정합니다.
    # val_end = int(len(df) * 0.85)
    val_end = 0

    # split 경계 계산 (시장 국면 기반)
    _dt_split = pd.to_datetime(df['datetime'])
    _train_end_row = int((_dt_split <= _PHASE_BULL_END).sum())
    _val_end_row   = int((_dt_split <= _PHASE_VAL_END).sum())

    # [Sync Fix 9] long/short 추론 피처 = OHLCV(5) + context(18) = 23 (훈련과 동일)
    test_pv_features  = df[price_vol_cols + context_cols].values.astype(np.float32)[val_end:]
    test_ctx_features = df[context_cols].values.astype(np.float32)[val_end:]
    test_close = raw_close[val_end:]
    test_dates = raw_dates[val_end:]

    logger.info(f"[INFO] 3개의 전문가 모델 로딩 중...")
    model_dir = os.path.join(ROOT_DIR, "checkpoints", "base_experts")

    # [Sync Fix 9/10] PriceActionExpert: input_dim=23, hidden_dim=128, dropout=0.4
    _pa_input_dim = len(price_vol_cols) + len(context_cols)
    long_model = PriceActionExpert(input_dim=_pa_input_dim, hidden_dim=128, dropout=0.4, use_attention=False).to(device)
    long_model.load_state_dict(torch.load(os.path.join(model_dir, "long_expert.pth"), map_location=device, weights_only=True))
    long_model.eval()

    short_model = PriceActionExpert(input_dim=_pa_input_dim, hidden_dim=128, dropout=0.4, use_attention=True).to(device)
    short_model.load_state_dict(torch.load(os.path.join(model_dir, "short_expert.pth"), map_location=device, weights_only=True))
    short_model.eval()

    context_model = ContextExpert(input_dim=len(context_cols)).to(device)
    context_model.load_state_dict(torch.load(os.path.join(model_dir, "context_expert.pth"), map_location=device, weights_only=True))
    context_model.eval()

    logger.info(f"[INFO] 배치 추론 시작...")
    pv_dataset = SlidingWindowDataset(test_pv_features, seq_length)
    ctx_dataset = SlidingWindowDataset(test_ctx_features, seq_length)
    
    pv_loader  = DataLoader(pv_dataset,  batch_size=batch_size, shuffle=False,
                            num_workers=get_optimal_workers(),
                            pin_memory=get_pin_memory(),
                            persistent_workers=(get_optimal_workers() > 0))
    ctx_loader = DataLoader(ctx_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=get_optimal_workers(),
                            pin_memory=get_pin_memory(),
                            persistent_workers=(get_optimal_workers() > 0))

    long_scores, short_scores, context_scores = [], [], []

    t0 = time.time()
    with torch.no_grad():
        for pv_batch, ctx_batch in zip(pv_loader, ctx_loader):
            pv_batch, ctx_batch = pv_batch.to(device), ctx_batch.to(device)
            
            # [Sync Fix 7] 전 expert logit 출력 → sigmoid로 확률 변환 후 저장
            # PriceActionExpert Sigmoid 제거(Fix 7)에 따라 long/short 도 sigmoid 필요
            long_scores.extend(torch.sigmoid(long_model(pv_batch)).cpu().numpy())
            short_scores.extend(torch.sigmoid(short_model(pv_batch)).cpu().numpy())
            context_scores.extend(torch.sigmoid(context_model(ctx_batch)).cpu().numpy())

    logger.info(f"[INFO] 추론 완료 ({time.time() - t0:.1f}초)")

    # [Sync Fix 1] 학습 시 end_idx가 seq_length-1부터 시작하므로
    # 추론 윈도우 idx=0의 마지막 캔들 = seq_length-1. 타임스탬프 오프셋 1 조정.
    out_start = seq_length - 1
    out_dates = test_dates[out_start : out_start + len(long_scores)]
    results_df = pd.DataFrame({
        'datetime': out_dates,
        'close': test_close[out_start : out_start + len(long_scores)],
        'long_score': np.round(long_scores, 4),
        'short_score': np.round(short_scores, 4),
        'context_score': np.round(context_scores, 4)
    })

    # split 메타 컬럼: 훈련/검증/테스트 구간을 명시해 in-sample 혼용을 방지합니다.
    _train_end_ts = raw_dates[_train_end_row] if _train_end_row < len(raw_dates) else None
    _val_end_ts   = raw_dates[_val_end_row]   if _val_end_row   < len(raw_dates) else None
    def _label_split(dt):
        if _train_end_ts is not None and dt <= _train_end_ts:
            return 'train'
        if _val_end_ts is not None and dt <= _val_end_ts:
            return 'val'
        return 'test'
    results_df['split'] = [_label_split(dt) for dt in out_dates]

    split_counts = results_df['split'].value_counts().to_dict()
    if split_counts.get('train', 0) > 0:
        logger.warning(
            f"[WARN] signals에 훈련 구간 행이 포함되어 있습니다: "
            f"train={split_counts.get('train',0):,}, "
            f"val={split_counts.get('val',0):,}, "
            f"test={split_counts.get('test',0):,}. "
            "RL 학습 시에는 split='val'+'test' (OOS) 행만 사용하세요."
        )

    output_dir = os.path.join(ROOT_DIR, "data", "signals")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    results_df.to_csv(out_path, index=False)
    logger.info(f"✅ 1, 2차 모델 점수 로그가 저장되었습니다: {out_path}")

    # RL 훈련용 표준 파일: val+test 구간 합산 → base_signals_log.csv
    # - val: 베이스 모델이 학습에 직접 사용하지 않은 OOS 구간 (조기종료 기준만 사용)
    # - test: 완전 OOS 구간
    # - train 구간 제외: 베이스 모델이 가중치를 업데이트한 in-sample이므로 신호가 과적합될 우려
    # BTC_USDT 기준 파일. 03_train_rl.py / 04_train_rl_batch.py의 기본 data-path와 일치.
    # 다른 코인은 {symbol}_signals_log.csv를 --data-path로 직접 지정하세요.
    _symbol_from_filename = output_filename.replace("_signals_log.csv", "")
    if _symbol_from_filename == "BTC_USDT":
        rl_df = results_df[results_df['split'].isin(['val', 'test'])].copy()
        rl_path = os.path.join(output_dir, "base_signals_log.csv")
        rl_df.to_csv(rl_path, index=False)
        val_cnt  = int((rl_df['split'] == 'val').sum())
        test_cnt = int((rl_df['split'] == 'test').sum())
        logger.info(
            f"✅ RL 훈련용 시그널 저장 (val+test): {rl_path}  "
            f"(rows={len(rl_df):,}  val={val_cnt:,}  test={test_cnt:,})"
        )
    else:
        logger.info(
            f"[INFO] base_signals_log.csv 미업데이트 (symbol={_symbol_from_filename}). "
            f"RL 훈련 시 --data-path data/signals/{output_filename} 로 지정하세요."
        )
    
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
    parser.add_argument("--symbol", type=str, default="BTC_USDT",
                        help="신호 추출 대상 심볼 (기본: BTC_USDT). 예: ETH_USDT, SOL_USDT, XRP_USDT")
    parser.add_argument("--data-path", type=str, default=None,
                        help="processed CSV 경로 (미지정 시 --symbol 기반 자동 생성)")
    parser.add_argument("--raw-data-path", type=str, default=None,
                        help="Raw CSV 경로 (지정 시 우선). 미지정 시 --data-path 기반 자동 생성")
    parser.add_argument("--seq-length", type=int, default=120,
                        help="입력 시퀀스 길이")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="추론 배치 크기")
    parser.add_argument("--output-filename", type=str, default=None,
                        help="data/signals 하위 출력 파일명 (미지정 시 {symbol}_signals_log.csv)")
    parser.add_argument(
        "--threshold-sets",
        type=str,
        default="",
        help=(
            "세미콜론 구분 임계치 세트. 형식: name,z_th,ctx_th,z_lead;... "
            "예: balanced_opt,0.0,0.49,0.0;conservative_opt,0.5,0.60,0.3"
        ),
    )
    args = parser.parse_args()

    # --symbol 기반 경로 자동 유도 (--data-path, --output-filename 명시 시 우선)
    if args.data_path is None:
        args.data_path = os.path.join(ROOT_DIR, "data", "processed", f"{args.symbol}_processed.csv")
    if args.output_filename is None:
        args.output_filename = f"{args.symbol}_signals_log.csv"

    if not os.path.exists(os.path.join(ROOT_DIR, "checkpoints", "base_experts", "long_expert.pth")):
        logger.error("[ERROR] Base 모델이 없습니다. 'python scripts/01_train_base.py'를 먼저 실행하세요.")
    else:
        logger.info(f"신호 추출 대상: {args.symbol} | 데이터: {args.data_path} | 출력: {args.output_filename}")
        extract_base_signals(
            data_path=args.data_path,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            threshold_sets=_parse_threshold_sets(args.threshold_sets),
            output_filename=args.output_filename,
            raw_data_path=args.raw_data_path,
        )