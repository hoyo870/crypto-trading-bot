"""
백테스트 배치 관리 스크립트.

역할:
  - tags.txt (또는 --tags 인자)에서 태그 목록을 읽어
    05_backtest.py 를 subprocess로 병렬 실행
  - 모든 백테스트 완료 후 06_rank.py 를 호출해 랭킹 산출
  - 결과를 best_by_leverage.csv 로 저장

단독 실행 예시:
  # tags.txt 자동 탐색 (--model-dir 하위)
  python scripts/07_backtest_batch.py \\
      --model-dir checkpoints/rl_generations/gen1 \\
      --reports-dir reports/gen1

  # 명시적 태그 지정
  python scripts/07_backtest_batch.py \\
      --tags lev1_stb_seed1234_001,lev1_bal_seed5678_001 \\
      --model-dir checkpoints/rl_generations/gen1 \\
      --reports-dir reports/gen1
"""

import os
import sys
import argparse
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 경로 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.platform_utils import get_optimal_jobs

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
logger = logging.getLogger("BacktestBatch")

_custom_log_dir = os.environ.get("CUSTOM_LOG_DIR")
if _custom_log_dir:
    os.makedirs(_custom_log_dir, exist_ok=True)
    logging.getLogger().addHandler(
        logging.FileHandler(os.path.join(_custom_log_dir, "backtest_batch.log"), encoding='utf-8')
    )

BACKTEST_SCRIPT = os.path.join(SCRIPT_DIR, "05_backtest.py")
RANK_SCRIPT     = os.path.join(SCRIPT_DIR, "06_rank.py")


# ── 모델 경로 탐색 ─────────────────────────────────────────────────────────
def _resolve_model_path(tag: str, model_dir: str):
    clean  = tag[:-4] if tag.endswith(".zip") else tag
    folder = os.path.join(model_dir, clean)
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.endswith(".zip") and ("final" in f or "best" in f):
                return os.path.join(folder, f)
    direct = os.path.join(model_dir, f"{clean}.zip")
    if os.path.exists(direct):
        return direct
    return None


# ── 단일 백테스트 subprocess 실행 ─────────────────────────────────────────
def _run_one(tag: str, model_dir: str, data_path: str,
             reports_dir: str, tuning_profile: str,
             env_type: str = "baby") -> tuple:
    model_path = _resolve_model_path(tag, model_dir)
    if not model_path:
        return tag, False, f"모델 파일 없음: {tag}"

    cmd = [
        sys.executable, BACKTEST_SCRIPT,
        "--model-path",     model_path,
        "--tag",            tag,
        "--data-path",      data_path,
        "--reports-dir",    reports_dir,
        "--tuning-profile", tuning_profile,
        "--env-type",       env_type,
    ]
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                                env=env, timeout=600)
        if result.returncode != 0:
            return tag, False, result.stderr.strip().splitlines()[-1]
        return tag, True, None
    except subprocess.TimeoutExpired:
        return tag, False, "타임아웃 (600s)"
    except Exception as exc:
        return tag, False, str(exc)


# ── 태그 목록 로드 ─────────────────────────────────────────────────────────
def _load_tags(tags_arg, model_dir: str) -> list:
    if tags_arg:
        return [t.strip() for t in tags_arg.split(",") if t.strip()]

    tags_file = os.path.join(model_dir, "tags.txt")
    if os.path.exists(tags_file):
        with open(tags_file, "r", encoding="utf-8") as f:
            tags = [line.strip() for line in f if line.strip()]
        logger.info(f"태그 목록 로드: {tags_file} ({len(tags)}개)")
        return tags

    tags = []
    if os.path.isdir(model_dir):
        for item in sorted(os.listdir(model_dir)):
            if item.startswith(".") or item == "tags.txt":
                continue
            if item.endswith(".zip"):
                tags.append(item[:-4])
            elif os.path.isdir(os.path.join(model_dir, item)):
                for f in os.listdir(os.path.join(model_dir, item)):
                    if f.endswith(".zip"):
                        tags.append(item)
                        break
    return sorted(set(tags))


# ── 배치 실행 ──────────────────────────────────────────────────────────────
def run_backtest_batch(tags: list, model_dir: str, data_path: str,
                       reports_dir: str, tuning_profile: str,
                       metric: str, formula: str = "balanced",
                       jobs: int = 5, env_type: str = "baby"):
    total = len(tags)
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"백테스트 배치 시작: {total}개 모델 (병렬 {jobs}개)")
    logger.info("=" * 70)

    os.makedirs(reports_dir, exist_ok=True)

    completed = 0
    success   = 0

    # 백테스트 병렬 실행 (subprocess → ThreadPoolExecutor 로 감싸 I/O 효율화)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _run_one, tag, model_dir, data_path, reports_dir, tuning_profile, env_type
            ): tag
            for tag in tags
        }

        for future in as_completed(futures):
            completed += 1
            tag_r, ok, err = future.result()
            if ok:
                success += 1
                logger.info(f"[{completed:03d}/{total}] 완료: {tag_r}")
            else:
                logger.error(f"[{completed:03d}/{total}] 실패: {tag_r} — {err}")

    logger.info(f"\n백테스트 완료: {success}/{total} 성공")

    # ── 랭킹 산출 (06_rank.py subprocess 호출) ────────────────────────────
    logger.info("\n랭킹 산출 시작...")
    rank_cmd = [
        sys.executable, RANK_SCRIPT,
        "--reports-dir", reports_dir,
        "--metric",      metric,
        "--formula",     formula,
    ]
    tags_file = os.path.join(model_dir, "tags.txt")
    if os.path.exists(tags_file):
        rank_cmd += ["--tags-file", tags_file]

    result = subprocess.run(rank_cmd, capture_output=False, text=True, encoding='utf-8',
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode != 0:
        logger.error(f"랭킹 산출 실패 (code={result.returncode})")
    else:
        logger.info("랭킹 산출 완료.")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="백테스트 배치 + 랭킹 (07)")

    default_model_dir   = os.path.join(ROOT_DIR, "checkpoints", "rl_generations")
    default_data_path   = os.path.join(ROOT_DIR, "data", "signals", "base_signals_log.csv")
    default_reports_dir = os.path.join(ROOT_DIR, "reports")

    parser.add_argument("--tags",         type=str, default=None,
                        help="쉼표 구분 태그 (미지정 시 tags.txt 또는 폴더 스캔)")
    parser.add_argument("--model-dir",    type=str, default=default_model_dir)
    parser.add_argument("--data-path",    type=str, default=default_data_path)
    parser.add_argument("--reports-dir",  type=str, default=default_reports_dir)
    parser.add_argument("--tuning-profile",
                        choices=["stable", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--env-type",
                        choices=["baby", "full"], default="baby",
                        help="백테스트 환경 (기본: baby). "
                             "finetune 모델 배치시 full 지정 필요.")
    parser.add_argument("--metric",
                        choices=["score", "total_return_pct", "sharpe_ratio", "mdd_pct"],
                        default="score")
    parser.add_argument("--formula",
                        choices=["balanced", "aggressive", "conservative"],
                        default="balanced",
                        help="점수 계산 공식 (--metric score일 때만 사용, 기본: balanced)")
    parser.add_argument("--jobs", type=int, default=get_optimal_jobs(),
                        help="병렬 프로세스 수 (기본: CPU코어//2 자동 감지)")

    args = parser.parse_args()

    # 환경변수 연동 (run_evolution.py 호환)
    args.model_dir   = os.environ.get("CUSTOM_MODEL_DIR",   args.model_dir)
    args.reports_dir = os.environ.get("CUSTOM_REPORTS_DIR", args.reports_dir)
    args.data_path   = os.environ.get("CUSTOM_DATA_PATH",   args.data_path)

    tags = _load_tags(args.tags, args.model_dir)
    if not tags:
        logger.error(f"처리할 태그 없음: {args.model_dir}")
        sys.exit(1)

    run_backtest_batch(
        tags=tags,
        model_dir=args.model_dir,
        data_path=args.data_path,
        reports_dir=args.reports_dir,
        tuning_profile=args.tuning_profile,
        metric=args.metric,
        formula=args.formula,
        jobs=args.jobs,
        env_type=args.env_type,
    )
