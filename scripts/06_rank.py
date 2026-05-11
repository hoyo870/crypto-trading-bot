"""
백테스트 결과 평가 및 랭킹 스크립트.

역할:
  - reports_dir 안의 summary_*.json 파일을 읽어 랭킹 산출
  - 레버리지별 베스트 모델을 best_by_leverage.csv 로 저장
  - 모델 목록(tags.txt)을 받아 처리하는 것도 지원

단독 실행 예시:
  # 디렉토리 내 JSON 자동 스캔
  python scripts/06_rank.py --reports-dir reports/gen1

  # tags.txt 기반 처리
  python scripts/06_rank.py \\
      --tags-file checkpoints/rl_generations/gen1/tags.txt \\
      --reports-dir reports/gen1
"""

import os
import sys
import json
import argparse
import logging

import pandas as pd

# ── 경로 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── 로깅 ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "orchestrator.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Rank")

_custom_log_dir = os.environ.get("CUSTOM_LOG_DIR")
if _custom_log_dir:
    os.makedirs(_custom_log_dir, exist_ok=True)
    logging.getLogger().addHandler(
        logging.FileHandler(os.path.join(_custom_log_dir, "rank.log"), encoding='utf-8')
    )


# ── 복합 점수 계산 ─────────────────────────────────────────────────────────
def _calc_score_balanced(s: dict) -> float:
    """
    균형 잡힌 공식 (기본): 수익성 + 효율성 - 위험성 - 청산페널티
    
    수익성(40%) + 샤프비(30%) - MDD절댓값(20%) + 승률(10%) - 청산패널티
    """
    ret     = float(s.get("total_return_pct", 0.0))
    sharpe  = float(s.get("sharpe_ratio", 0.0))
    mdd     = abs(float(s.get("mdd_pct", 0.0)))
    wr      = float(s.get("win_rate", 0.0))
    liq     = 1.0 if s.get("liquidated", False) else 0.0
    
    # 승률이 0~1 범위이면 백분율로 변환, 0~100 범위면 그대로 사용
    wr_pct = wr * 100 if wr <= 1 else wr
    
    return (
        ret * 0.40
        + sharpe * 8.0 * 0.30
        - mdd * 0.20
        + (wr_pct / 100) * 50 * 0.10
        - liq * 200.0
    )


def _calc_score_aggressive(s: dict) -> float:
    """
    공격적 공식: 수익성 최우선, 위험도 경고만
    
    수익성을 크게 가중치주고, 청산이면 강력한 페널티
    """
    ret     = float(s.get("total_return_pct", 0.0))
    sharpe  = float(s.get("sharpe_ratio", 0.0))
    mdd     = abs(float(s.get("mdd_pct", 0.0)))
    liq     = 1.0 if s.get("liquidated", False) else 0.0
    
    return (
        ret * 1.0
        + sharpe * 5.0
        - mdd * 0.3
        - liq * 500.0
    )


def _calc_score_conservative(s: dict) -> float:
    """
    보수적 공식: 위험도 최우선, 안정성 강조
    
    MDD와 청산을 크게 가중치주고, 수익성은 보조
    """
    ret     = float(s.get("total_return_pct", 0.0))
    sharpe  = float(s.get("sharpe_ratio", 0.0))
    mdd     = abs(float(s.get("mdd_pct", 0.0)))
    wr      = float(s.get("win_rate", 0.0))
    liq     = 1.0 if s.get("liquidated", False) else 0.0
    
    wr_pct = wr * 100 if wr <= 1 else wr
    
    return (
        ret * 0.30
        + sharpe * 10.0
        - mdd * 0.60
        + (wr_pct / 100) * 30
        - liq * 1000.0
    )


def _calc_score(s: dict, formula: str = "balanced") -> float:
    """
    여러 공식 중 선택하여 종합 점수 계산
    
    Args:
        s: 모델 결과 dict (total_return_pct, sharpe_ratio, mdd_pct, liquidated 등)
        formula: "balanced" | "aggressive" | "conservative"
    
    Returns:
        종합 점수 (높을수록 좋음)
    """
    if formula == "aggressive":
        return _calc_score_aggressive(s)
    elif formula == "conservative":
        return _calc_score_conservative(s)
    else:  # balanced (기본)
        return _calc_score_balanced(s)


def _metric_value(s: dict, metric: str, formula: str = "balanced") -> float:
    """단일 지표 또는 종합 점수 반환"""
    if metric == "total_return_pct":
        return float(s.get("total_return_pct", float("-inf")))
    elif metric == "sharpe_ratio":
        return float(s.get("sharpe_ratio", float("-inf")))
    elif metric == "mdd_pct":
        # MDD는 음수이므로 작을수록(음수가 클수록) 좋음
        # 정렬을 위해 음수 반환
        return -float(s.get("mdd_pct", 0.0))
    else:  # "score" 또는 기타
        return _calc_score(s, formula=formula)


# ── JSON 로더 ──────────────────────────────────────────────────────────────
def _load_summaries(reports_dir: str, tags: list | None = None) -> list[dict]:
    """
    reports_dir 에서 summary_*.json 파일을 읽어 dict 목록으로 반환합니다.
    tags 목록이 주어지면 해당 태그만 필터링합니다.
    """
    summaries = []
    if not os.path.isdir(reports_dir):
        logger.warning(f"reports_dir 없음: {reports_dir}")
        return summaries

    tag_set = set(tags) if tags else None

    for fname in sorted(os.listdir(reports_dir)):
        if not (fname.startswith("summary_") and fname.endswith(".json")):
            continue
        tag = fname[len("summary_"):-len(".json")]
        if tag_set is not None and tag not in tag_set:
            continue
        path = os.path.join(reports_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # tag 필드가 없으면 파일명으로 보완
            data.setdefault("tag", tag)
            summaries.append(data)
        except Exception as e:
            logger.warning(f"JSON 읽기 실패 ({fname}): {e}")

    return summaries


# ── 랭킹 산출 ──────────────────────────────────────────────────────────────
def rank_results(summaries: list[dict], metric: str = "score",
                 formula: str = "balanced", reports_dir: str = ".") -> pd.DataFrame:
    """
    summaries 목록을 metric 기준으로 정렬하고,
    전체 모델 랭킹을 best_by_leverage.csv 로 저장합니다.

    Args:
        summaries: 백테스트 결과 dict 목록
        metric: "score" | "total_return_pct" | "sharpe_ratio" | "mdd_pct"
        formula: "balanced" | "aggressive" | "conservative" (metric="score"일 때만 사용)
        reports_dir: CSV 저장 경로

    Returns:
        전체 랭킹 DataFrame
    """
    if not summaries:
        logger.warning("랭킹 산출 대상 데이터 없음.")
        return pd.DataFrame()

    # 복합 점수 컬럼 추가
    for s in summaries:
        s["_score"] = _metric_value(s, metric, formula=formula)

    df = pd.DataFrame(summaries)
    df = df.sort_values("_score", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based 랭킹

    # 전체 랭킹 로그
    logger.info("")
    logger.info(f"{'='*70}")
    formula_label = f"{metric} (공식: {formula})" if metric == "score" else metric
    logger.info(f"🏅 전체 랭킹 (기준: {formula_label})")
    logger.info(f"{'='*70}")
    for rank, row in df.iterrows():
        liq_mark = "⚠️청산" if row.get("liquidated") else ""
        logger.info(
            f"[{rank:03d}] {row['tag']:<50} "
            f"수익률={row.get('total_return_pct', 0):+7.2f}% | "
            f"MDD={row.get('mdd_pct', 0):5.2f}% | "
            f"Sharpe={row.get('sharpe_ratio', 0):.3f} | "
            f"점수={row.get('_score', 0):.1f} {liq_mark}"
        )

    # 레버리지별 베스트 (로그 출력용)
    grouped = df.groupby("leverage")
    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"🏆 레버리지별 베스트 (기준: {metric})")
    logger.info(f"{'='*70}")
    for lev, grp in sorted(grouped, key=lambda x: x[0]):
        best = grp.iloc[0]
        logger.info(
            f"[Lev {lev}x] {best['tag']} | "
            f"수익률={best.get('total_return_pct', 0):+.2f}% | "
            f"MDD={best.get('mdd_pct', 0):.2f}% | "
            f"승률={best.get('win_rate', 0):.1f}%"
        )

    # ✅ 전체 90개 모델 랭킹을 CSV에 저장 (auto_discard_models에서 keep_top_k 적용 가능)
    output_rows = []
    for rank, row in df.iterrows():
        output_rows.append({
            "rank":             rank,
            "tag":              row["tag"],
            "leverage":         row.get("leverage", 0),
            "metric":           metric,
            "metric_value":     row["_score"],
            "total_return_pct": row.get("total_return_pct", 0.0),
            "mdd_pct":          row.get("mdd_pct", 0.0),
            "sharpe_ratio":     row.get("sharpe_ratio", 0.0),
            "liquidated":       row.get("liquidated", False),
        })

    output_df = pd.DataFrame(output_rows)
    out_path = os.path.join(reports_dir, "best_by_leverage.csv")
    output_df.to_csv(out_path, index=False)
    logger.info(f"\n💾 전체 모델 랭킹 저장: {out_path} ({len(output_df)}개)")

    return df.drop(columns=["_score"])


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="백테스트 결과 평가/랭킹",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
[점수 공식 설명]
  
  balanced (기본, 균형 잡힌 공식):
    수익률(40%) + 샤프비(30%) - MDD(20%) + 승률(10%) - 청산페널티
    → 안정성과 수익성의 균형을 추구하는 전형적인 트레이더 관점
    
  aggressive (공격적 공식):
    수익률(100%) + 샤프비(50%) - MDD(30%) - 청산페널티(500)
    → 고수익 추구, 위험도는 경고 수준만 체크
    → 청산되지 않는 한 위험도 크게 신경쓰지 않음
    
  conservative (보수적 공식):
    수익률(30%) + 샤프비(100%) - MDD(60%) + 승률(30%) - 청산페널티(1000)
    → 안정성 최우선, 최대 낙폭과 승률을 강조
    → 청산되는 경우 매우 큰 페널티

[사용 예시]
  # balanced 공식으로 랭킹 (기본)
  python scripts/06_rank.py --reports-dir reports/gen1
  
  # aggressive 공식으로 랭킹
  python scripts/06_rank.py --reports-dir reports/gen1 --metric score --formula aggressive
  
  # conservative 공식으로 랭킹
  python scripts/06_rank.py --reports-dir reports/gen1 --metric score --formula conservative
  
  # 순수 수익률 기준 랭킹
  python scripts/06_rank.py --reports-dir reports/gen1 --metric total_return_pct
        """
    )

    parser.add_argument("--reports-dir", type=str,
                        default=os.path.join(ROOT_DIR, "reports"),
                        help="summary_*.json 파일이 있는 디렉토리")
    parser.add_argument("--tags-file", type=str, default=None,
                        help="처리할 태그 목록 파일 (tags.txt). 미지정 시 디렉토리 내 전체 스캔")
    parser.add_argument("--metric",
                        choices=["score", "total_return_pct", "sharpe_ratio", "mdd_pct"],
                        default="score",
                        help="랭킹 기준 지표 (기본: score)")
    parser.add_argument("--formula",
                        choices=["balanced", "aggressive", "conservative"],
                        default="balanced",
                        help="점수 계산 공식 (--metric score일 때만 사용, 기본: balanced)")

    args = parser.parse_args()

    # 환경변수 연동
    args.reports_dir = os.environ.get("CUSTOM_REPORTS_DIR", args.reports_dir)

    tags = None
    if args.tags_file and os.path.exists(args.tags_file):
        with open(args.tags_file, "r", encoding="utf-8") as f:
            tags = [line.strip() for line in f if line.strip()]
        logger.info(f"태그 목록 로드: {args.tags_file} ({len(tags)}개)")

    summaries = _load_summaries(args.reports_dir, tags)
    if not summaries:
        logger.error(f"처리할 결과 없음: {args.reports_dir}")
        sys.exit(1)

    logger.info(f"로드된 결과: {len(summaries)}개")
    rank_results(summaries, metric=args.metric, formula=args.formula, reports_dir=args.reports_dir)
