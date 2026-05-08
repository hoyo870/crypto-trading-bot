import os
import sys
import subprocess
import random
import time
import argparse
from collections import deque
from itertools import product

# 현재 스크립트가 있는 디렉토리
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 기본 병렬 수 (안정성/속도 균형)
DEFAULT_PARALLEL_JOBS = 3

DEFAULT_PROFILES = ("stable", "balanced", "aggressive")
DEFAULT_LEVERAGES = (1, 1, 1)
DEFAULT_COUNT_PER_TASK = 10


def _resolve_latest_tags_for_seeds(candidates_dir, leverage, seeds):
    resolved = []
    if not os.path.isdir(candidates_dir):
        return resolved

    for seed in seeds:
        prefix = f"lev{int(leverage)}_seed{int(seed)}_"
        max_idx = -1
        best_tag = None
        for f in os.listdir(candidates_dir):
            if not f.endswith(".zip"):
                continue
            stem = f[:-4]
            if not stem.startswith(prefix):
                continue
            suffix = stem[len(prefix):]
            if suffix.isdigit() and len(suffix) == 3:
                idx = int(suffix)
                if idx > max_idx:
                    max_idx = idx
                    best_tag = stem
        if best_tag is not None:
            resolved.append(best_tag)
    return resolved

def _parse_csv_ints(raw):
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_csv_strs(raw):
    return [x.strip().lower() for x in str(raw).split(",") if x.strip()]


def _build_tasks(leverages, profiles, count_per_task):
    # 실행할 (count, leverage, profile) 작업 목록
    # count_per_task=10이면 (1개 학습 + 즉시 백테스트) 작업 10개를 생성
    tasks = []
    for lev, profile in product(leverages, profiles):
        for _ in range(count_per_task):
            tasks.append((1, lev, profile))
    return tasks


def _launch_training(count, leverage, tuning_profile):
    seeds = [random.randint(0, 10000) for _ in range(count)]
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "run_train.py"),
        "--count", str(count),
        "--leverage", str(leverage),
        "--tuning-profile", tuning_profile,
        "--top-k", "0",
        "--seeds", ",".join(str(s) for s in seeds),
        "--no-improve-start-ratio", "0.1",  # 학습 초반 10%는 no-improve 카운트 시작 안 함
        "--split-mode", "holdout",
    ]
    return subprocess.Popen(cmd, cwd=BASE_DIR), seeds


def _run_backtest_after_training(tags, leverage, tuning_profile):
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "run_backtest.py"),
        "--tags", ",".join(tags),
        "--source", "candidates",
        "--workers", "1",
        "--leverage", str(leverage),
        "--tuning-profile", tuning_profile,
    ]
    print(f"\n{'='*60}")
    print(f"📊 [후처리] run_backtest.py 자동 실행 시작 | tags={tags}")
    print(f"{'='*60}")

    ret = subprocess.run(cmd, cwd=BASE_DIR)
    if ret.returncode != 0:
        print(f"[ERROR] run_backtest.py 실행 실패 (Exit code: {ret.returncode})")
        sys.exit(ret.returncode)

    print("[SUCCESS] run_backtest.py 실행 완료")


def run_parallel_trainings(parallel_jobs=DEFAULT_PARALLEL_JOBS,
                          leverages=None,
                          profiles=None,
                          count_per_task=DEFAULT_COUNT_PER_TASK,
                          run_backtest=True):
    if leverages is None:
        leverages = list(DEFAULT_LEVERAGES)
    if profiles is None:
        profiles = list(DEFAULT_PROFILES)

    tasks = deque(_build_tasks(leverages=leverages,
                               profiles=profiles,
                               count_per_task=count_per_task))
    if not tasks:
        print("[ERROR] 실행할 작업이 없습니다. --leverages / --profiles / --count-per-task 확인 필요")
        sys.exit(1)

    parallel_jobs = max(1, min(int(parallel_jobs), len(tasks)))
    active = []

    print(f"\n{'='*60}")
    print("🚀 [배치 자동화] 레버리지별 병렬 비동기 학습 파이프라인 가동 시작")
    print(f"[INFO] 동시 실행 수: {parallel_jobs}")
    print(f"[INFO] 레버리지 목록: {leverages}")
    print(f"[INFO] 프로파일 목록: {profiles}")
    print(f"[INFO] 작업당 모델 수: {count_per_task}")
    print(f"{'='*60}")

    while tasks or active:
        while tasks and len(active) < parallel_jobs:
            count, leverage, profile = tasks.popleft()
            print(f"\n{'*'*50}")
            print(f"▶️ 실행 시작: 레버리지 {leverage}x | profile={profile} | 모델 {count}개")
            print(f"{'*'*50}\n")
            proc, seeds = _launch_training(count, leverage, profile)

            active.append((proc, count, leverage, profile, seeds))

        for proc, count, leverage, profile, seeds in active[:]:
            ret = proc.poll()
            if ret is None:
                continue

            active.remove((proc, count, leverage, profile, seeds))

            if ret != 0:
                print(
                    f"\n[ERROR] ❌ 레버리지 {leverage}x | profile={profile} "
                    f"훈련 중 치명적인 에러 발생! (Exit code: {ret})"
                )
                print("[INFO] 안전을 위해 실행 중인 나머지 훈련을 모두 중단합니다.")
                for other_proc, _, _, _, _ in active:
                    if other_proc.poll() is None:
                        other_proc.terminate()
                for other_proc, _, _, _, _ in active:
                    other_proc.wait()
                sys.exit(ret)

            print(f"\n[SUCCESS] ✅ 레버리지 {leverage}x | profile={profile} 훈련 무사히 완료!")

            if run_backtest:
                candidates_dir = os.path.join(os.path.dirname(BASE_DIR), "models", "commander", "candidates")
                resolved_tags = _resolve_latest_tags_for_seeds(candidates_dir, leverage, seeds)
                if not resolved_tags:
                    print("[WARN] 학습 직후 백테스트 대상 태그를 찾지 못해 이번 건은 건너뜁니다.")
                else:
                    _run_backtest_after_training(resolved_tags, leverage, profile)

        # 과도한 busy-loop 방지
        if active:
            time.sleep(1)

    print(f"\n{'='*60}")
    print("🎉 모든 v6 병렬 학습이 완벽하게 종료되었습니다!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="레버리지별 병렬 학습 실행기")
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL_JOBS,
        help=f"동시 실행할 학습 프로세스 수 (기본: {DEFAULT_PARALLEL_JOBS})"
    )
    parser.add_argument(
        "--leverages",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LEVERAGES),
        help="쉼표 구분 레버리지 목록 (예: 1,3,5)"
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=",".join(DEFAULT_PROFILES),
        help="쉼표 구분 튜닝 프로파일 목록 (stable,balanced,aggressive)"
    )
    parser.add_argument(
        "--count-per-task",
        type=int,
        default=DEFAULT_COUNT_PER_TASK,
        help="각 (레버리지,프로파일) 조합에서 학습할 모델 수 (각 모델은 개별 프로세스로 학습 후 즉시 백테스트)"
    )
    parser.add_argument(
        "--run-backtest",
        dest="run_backtest",
        action="store_true",
        help="각 개별 학습 완료 직후 run_backtest.py 자동 실행 (기본: true)"
    )
    parser.add_argument(
        "--no-run-backtest",
        dest="run_backtest",
        action="store_false",
        help="개별 학습 완료 직후 run_backtest.py 자동 실행 비활성화"
    )
    parser.set_defaults(run_backtest=True)
    args = parser.parse_args()

    profiles = _parse_csv_strs(args.profiles)
    invalid_profiles = [p for p in profiles if p not in {"stable", "balanced", "aggressive"}]
    if invalid_profiles:
        print(f"[ERROR] 알 수 없는 프로파일: {invalid_profiles}")
        sys.exit(1)

    run_parallel_trainings(
        parallel_jobs=args.parallel,
        leverages=_parse_csv_ints(args.leverages),
        profiles=profiles,
        count_per_task=max(1, int(args.count_per_task)),
        run_backtest=args.run_backtest,
    )