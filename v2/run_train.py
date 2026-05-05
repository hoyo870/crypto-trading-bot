"""
run_train.py — RL Commander 모델 5개 순차 학습 스크립트

각 학습은 서로 다른 seed를 사용해 다양성을 확보하며,
SmartStopCallback 3가지 조건(Early Stop / 퇴화 감지 / 목표 달성)으로
불필요한 학습을 자동 종료함.

학습 설정:
  - total_timesteps : 3,000,000 (모델당)
  - eval_freq       : 10,000
  - patience        : 10
  - reward_target   : 99
  - entropy_threshold: -0.01
  - seeds           : 1, 7, 42, 123, 777
"""
import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from train_rl_commander import train_commander

# ── 학습 설정 ────────────────────────────────────────────────────
TOTAL_TIMESTEPS    = 3_000_000
EVAL_FREQ          = 10_000
PATIENCE           = 10
REWARD_TARGET      = 99.0   # 목표 달성 시 해당 모델 즉시 종료
ENTROPY_THRESHOLD  = -0.01  # 정책 퇴화 감지 임계값

# 모델별 seed (다양한 초기화로 탐색 다양성 확보)
SEEDS = [1, 7, 42, 123, 777]
# ────────────────────────────────────────────────────────────────


def main():
    total = len(SEEDS)
    results = []

    for idx, seed in enumerate(SEEDS, start=1):
        print(f"\n{'#'*60}")
        print(f"# 모델 {idx}/{total} 학습 시작  (seed={seed})")
        print(f"{'#'*60}")
        try:
            train_commander(
                total_timesteps=TOTAL_TIMESTEPS,
                eval_freq=EVAL_FREQ,
                patience=PATIENCE,
                reward_target=REWARD_TARGET,
                entropy_threshold=ENTROPY_THRESHOLD,
                seed=seed,
                model_tag=None,   # 자동 mNNN 태그 배정
            )
            results.append((seed, "OK"))
        except Exception as e:
            print(f"\n[ERROR] seed={seed} 학습 중 예외 발생:")
            traceback.print_exc()
            results.append((seed, f"FAIL: {e}"))

    print(f"\n{'='*60}")
    print("학습 완료 요약")
    print(f"{'='*60}")
    for seed, status in results:
        print(f"  seed={seed:>4}  →  {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
