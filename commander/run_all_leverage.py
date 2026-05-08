import os
import sys
import subprocess
import random
import time
import argparse
from collections import deque

# 현재 스크립트가 있는 디렉토리
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# M1 Max 32GB 기준 기본 병렬 수 (안정성/속도 균형)
DEFAULT_PARALLEL_JOBS = 3

def _build_tasks():
    # 실행할 (count, leverage) 작업 목록
    return [
        (10, 1),
        (10, 3),
        (10, 5)
    ]


def _launch_training(count, leverage):
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "run_train.py"),
        "--count", str(count),
        "--leverage", str(leverage),
        "--seeds", ",".join(str(random.randint(0, 10000)) for _ in range(count)),
        "--no-improve-start-ratio", "0.2",  # 학습 초반 20%는 no-improve 카운트 시작 안 함
    ]
    return subprocess.Popen(cmd, cwd=BASE_DIR)


def run_parallel_trainings(parallel_jobs=DEFAULT_PARALLEL_JOBS):
    tasks = deque(_build_tasks())
    parallel_jobs = max(1, min(int(parallel_jobs), len(tasks)))
    active = []

    print(f"\n{'='*60}")
    print("🚀 [배치 자동화] 레버리지별 병렬 비동기 학습 파이프라인 가동 시작")
    print(f"[INFO] 동시 실행 수: {parallel_jobs}")
    print(f"{'='*60}")

    while tasks or active:
        while tasks and len(active) < parallel_jobs:
            count, leverage = tasks.popleft()
            print(f"\n{'*'*50}")
            print(f"▶️ 실행 시작: 레버리지 {leverage}x (시드 {count}개)")
            print(f"{'*'*50}\n")
            proc = _launch_training(count, leverage)
            active.append((proc, count, leverage))

        for proc, count, leverage in active[:]:
            ret = proc.poll()
            if ret is None:
                continue

            active.remove((proc, count, leverage))

            if ret != 0:
                print(f"\n[ERROR] ❌ 레버리지 {leverage}x 훈련 중 치명적인 에러 발생! (Exit code: {ret})")
                print("[INFO] 안전을 위해 실행 중인 나머지 훈련을 모두 중단합니다.")
                for other_proc, _, _ in active:
                    if other_proc.poll() is None:
                        other_proc.terminate()
                for other_proc, _, _ in active:
                    other_proc.wait()
                sys.exit(ret)

            print(f"\n[SUCCESS] ✅ 레버리지 {leverage}x 훈련 무사히 완료!")

        # 과도한 busy-loop 방지
        if active:
            time.sleep(1)

    print(f"\n{'='*60}")
    print("🎉 모든 병렬 학습 (1x, 3x, 5x)이 완벽하게 종료되었습니다!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="레버리지별 병렬 학습 실행기")
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL_JOBS,
        help=f"동시 실행할 학습 프로세스 수 (기본: {DEFAULT_PARALLEL_JOBS})"
    )
    args = parser.parse_args()

    run_parallel_trainings(parallel_jobs=args.parallel)