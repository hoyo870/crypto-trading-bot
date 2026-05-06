"""
Commander 일괄 백테스트 실행기

사용법:
  python run_backtest.py                          # 기본(전체 candidates, leverage=2)
  python run_backtest.py --leverage 3             # 3배 레버리지
    python run_backtest.py --tags lev2_seed42_001,lev2_seed43_001 --leverage 2
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def run_backtest_all(tags, leverage, model_dir, log_dir, data_path, reports_dir):
    os.makedirs(log_dir, exist_ok=True)

    results = []
    for i, tag in enumerate(tags, start=1):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"backtest_{i:02d}_{tag}_{ts}.log")

        print(f"\n[{i:02d}/{len(tags)}] 백테스트: tag={tag} | leverage={leverage}x ...")
        cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "backtest_rl_commander.py"),
            "--model-tag", tag,
            "--suffix", tag,
            "--leverage", str(leverage),
            "--model-dir", model_dir,
            "--data-path", data_path,
            "--reports-dir", reports_dir,
        ]

        with open(log_path, "w") as lf:
            ret = subprocess.run(cmd, capture_output=False,
                                 stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=BASE_DIR)

        status = "OK" if ret.returncode == 0 else "FAIL"
        results.append((tag, status, log_path))
        print(f"[{'DONE' if status == 'OK' else 'ERROR'}] tag={tag} -> {status}")

    print("\n" + "=" * 60)
    print("백테스트 완료 요약")
    print("=" * 60)
    for tag, status, log_path in results:
        print(f"  tag={tag}  ->  {status}  | log={log_path}")
    print("=" * 60)


def _auto_discover_tags(model_dir):
    candidates_dir = os.path.join(model_dir, "candidates")
    if not os.path.isdir(candidates_dir):
        return []
    tags = sorted(
        f[:-4] for f in os.listdir(candidates_dir)
        if f.endswith(".zip") and not f.startswith(".")
    )
    return tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commander 일괄 백테스트")
    parser.add_argument("--tags", type=str, default=None,
                        help="쉼표 구분 태그 (미지정 시 candidates 전체)")
    parser.add_argument("--leverage", type=int, default=2,
                        help="레버리지 배수 (기본 2)")
    parser.add_argument("--model-dir", type=str,
                        default=os.path.join(ROOT_DIR, "models", "commander"),
                        help="모델 루트 디렉토리")
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join(BASE_DIR, "logs", "backtest"),
                        help="백테스트 로그 디렉토리")
    parser.add_argument("--data-path", type=str,
                        default=os.path.join(ROOT_DIR, "data", "commander", "base_signals_log.csv"),
                        help="백테스트 데이터 CSV")
    parser.add_argument("--reports-dir", type=str,
                        default=os.path.join(BASE_DIR, "reports"),
                        help="백테스트 결과 리포트 디렉토리")
    args = parser.parse_args()

    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = _auto_discover_tags(args.model_dir)
        if not tags:
            print("[ERROR] candidates 폴더에 모델이 없습니다.")
            sys.exit(1)
        print(f"[INFO] 자동 발견된 모델 {len(tags)}개: {tags}")

    run_backtest_all(
        tags=tags,
        leverage=args.leverage,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        data_path=args.data_path,
        reports_dir=args.reports_dir,
    )
