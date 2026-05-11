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
def _calc_score(s: dict) -> float:
    """
    total_return + sharpe*10 - mdd*0.5 - 청산패널티

    metric="score" 기준이며, 다른 기준 선택 시 해당 값만 사용.
    """
    liq_penalty = 1000.0 if s.get("liquidated", False) else 0.0
    return (
        float(s.get("total_return_pct", 0.0))
        + float(s.get("sharpe_ratio", 0.0)) * 10.0
        - abs(float(s.get("mdd_pct", 0.0))) * 0.5
        - liq_penalty
    )


def _metric_value(s: dict, metric: str) -> float:
    if metric == "total_return_pct":
        return float(s.get("total_return_pct", float("-inf")))
    if metric == "sharpe_ratio":
        return float(s.get("sharpe_ratio", float("-inf")))
    if metric == "mdd_pct":
        return float(s.get("mdd_pct", float("-inf")))
    return _calc_score(s)  # "score" (기본)


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
                 reports_dir: str = ".") -> pd.DataFrame:
    """
    summaries 목록을 metric 기준으로 정렬하고,
    레버리지별 베스트 결과를 best_by_leverage.csv 로 저장합니다.

    Returns:
        전체 랭킹 DataFrame
    """
    if not summaries:
        logger.warning("랭킹 산출 대상 데이터 없음.")
        return pd.DataFrame()

    # 복합 점수 컬럼 추가
    for s in summaries:
        s["_score"] = _metric_value(s, metric)

    df = pd.DataFrame(summaries)
    df = df.sort_values("_score", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based 랭킹

    # 전체 랭킹 로그
    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"🏅 전체 랭킹 (기준: {metric})")
    logger.info(f"{'='*70}")
    for rank, row in df.iterrows():
        liq_mark = "⚠️청산" if row.get("liquidated") else ""
        logger.info(
            f"[{rank:03d}] {row['tag']:<50} "
            f"수익률={row.get('total_return_pct', 0):+7.2f}% | "
            f"MDD={row.get('mdd_pct', 0):5.2f}% | "
            f"Sharpe={row.get('sharpe_ratio', 0):.3f} {liq_mark}"
        )

    # 레버리지별 베스트
    grouped = df.groupby("leverage")
    best_rows = []
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
        best_rows.append({
            "leverage":         lev,
            "tag":              best["tag"],
            "metric":           metric,
            "metric_value":     best["_score"],
            "total_return_pct": best.get("total_return_pct", 0.0),
            "mdd_pct":          best.get("mdd_pct", 0.0),
            "sharpe_ratio":     best.get("sharpe_ratio", 0.0),
            "liquidated":       best.get("liquidated", False),
        })

    best_df = pd.DataFrame(best_rows)
    out_path = os.path.join(reports_dir, "best_by_leverage.csv")
    best_df.to_csv(out_path, index=False)
    logger.info(f"\n💾 best_by_leverage.csv 저장: {out_path}")

    return df.drop(columns=["_score"])


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="백테스트 결과 평가/랭킹")

    parser.add_argument("--reports-dir", type=str,
                        default=os.path.join(ROOT_DIR, "reports"),
                        help="summary_*.json 파일이 있는 디렉토리")
    parser.add_argument("--tags-file", type=str, default=None,
                        help="처리할 태그 목록 파일 (tags.txt). 미지정 시 디렉토리 내 전체 스캔")
    parser.add_argument("--metric",
                        choices=["score", "total_return_pct", "sharpe_ratio", "mdd_pct"],
                        default="score",
                        help="랭킹 기준 지표 (기본: score)")

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
    rank_results(summaries, metric=args.metric, reports_dir=args.reports_dir)
