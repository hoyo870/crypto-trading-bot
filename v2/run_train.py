"""RL Commander 모델 5개 순차 학습 실행기.

- 각 모델의 전체 stdout/stderr를 파일로 저장
- 터미널에는 SB3 표형 블록 로그를 제외한 핵심 로그만 출력
"""
import os
import sys
import re
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── 학습 설정 ────────────────────────────────────────────────────
TOTAL_TIMESTEPS    = 3_000_000
EVAL_FREQ          = 10_000
PATIENCE           = 20
REWARD_TARGET      = 99.0   # 목표 달성 시 해당 모델 즉시 종료
ENTROPY_THRESHOLD  = -0.01  # 정책 퇴화 감지 임계값

# 모델별 seed (다양한 초기화로 탐색 다양성 확보)
SEEDS = [1, 7, 42, 123, 777, 2024, 9999, 12345, 54321, 99999]
# ────────────────────────────────────────────────────────────────


def _is_block_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    # SB3 표 블록 경계선: -----------------------------
    if re.fullmatch(r"-+", stripped) and len(stripped) >= 10:
        return True
    # SB3 표 본문: | eval/... |
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    return False


def _run_one(seed: int, idx: int, total: int, logs_dir: str) -> tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"train_{idx:02d}_seed{seed}_{ts}.log"
    log_path = os.path.join(logs_dir, log_name)

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "train_rl_commander.py"),
        "--total-timesteps", str(TOTAL_TIMESTEPS),
        "--eval-freq", str(EVAL_FREQ),
        "--patience", str(PATIENCE),
        "--reward-target", str(REWARD_TARGET),
        "--entropy-threshold", str(ENTROPY_THRESHOLD),
        "--seed", str(seed),
    ]

    print(f"\n{'#'*60}")
    print(f"# 모델 {idx}/{total} 학습 시작 (seed={seed})")
    print(f"# 로그 파일: {log_path}")
    print(f"{'#'*60}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("[COMMAND] " + " ".join(cmd) + "\n\n")

        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            if not _is_block_line(line):
                print(line, end="")

        code = proc.wait()

    status = "OK" if code == 0 else f"FAIL(exit={code})"
    print(f"[DONE] seed={seed} -> {status}")
    return status, log_path


def main():
    logs_dir = os.path.join(BASE_DIR, "logs", "train")
    os.makedirs(logs_dir, exist_ok=True)

    total = len(SEEDS)
    results = []

    for idx, seed in enumerate(SEEDS, start=1):
        status, log_path = _run_one(seed=seed, idx=idx, total=total, logs_dir=logs_dir)
        results.append((seed, status, log_path))

    print(f"\n{'='*60}")
    print("학습 완료 요약")
    print(f"{'='*60}")
    for seed, status, log_path in results:
        print(f"  seed={seed:>4}  →  {status}  | log={log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
