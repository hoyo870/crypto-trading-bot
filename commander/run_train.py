"""
Commander 일괄 학습 실행기 (통합 레버리지 환경)

예시:
  python run_train.py --count 3 --leverage 2 --improved-hp
  python run_train.py --count 1 --leverage 3 --timesteps 3000000
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _next_tag(candidates_dir):
    os.makedirs(candidates_dir, exist_ok=True)
    max_idx = 0
    for f in os.listdir(candidates_dir):
        if f.startswith("m") and f.endswith(".zip") and len(f) == 8:
            try:
                max_idx = max(max_idx, int(f[1:4]))
            except ValueError:
                pass
    return f"m{max_idx + 1:03d}"


def _resolve_seeds(count, base_seed, seed_step, seeds_csv):
    if seeds_csv:
        seeds = []
        for tok in seeds_csv.split(","):
            tok = tok.strip()
            if not tok:
                continue
            seeds.append(int(tok))
        if len(seeds) < count:
            raise ValueError(
                f"--seeds 개수({len(seeds)})가 --count({count})보다 작습니다."
            )
        return seeds[:count]

    return [base_seed + i * seed_step for i in range(count)]


def run_train_batch(count, leverage, timesteps, patience, improved_hp,
                    base_seed, seed_step, seeds_csv,
                    model_dir, log_dir, data_path):
    os.makedirs(log_dir, exist_ok=True)
    candidates_dir = os.path.join(model_dir, "candidates")

    seeds = _resolve_seeds(
        count=count,
        base_seed=base_seed,
        seed_step=seed_step,
        seeds_csv=seeds_csv,
    )
    print(f"[INFO] 사용 시드 목록({len(seeds)}): {seeds}")

    results = []
    for i in range(count):
        tag = _next_tag(candidates_dir)
        seed = seeds[i]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"train_{tag}_seed{seed}_{ts}.log")

        cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "train_rl_commander.py"),
            "--tag", tag,
            "--leverage", str(leverage),
            "--total-timesteps", str(timesteps),
            "--patience", str(patience),
            "--reward-target", "1000000000",
            "--seed", str(seed),
            "--model-dir", model_dir,
            "--data-path", data_path,
        ]
        if improved_hp:
            cmd.append("--improved-hp")

        print(f"\n[{i+1}/{count}] 학습 시작: tag={tag}, lev={leverage}x, seed={seed}")
        with open(log_path, "w") as lf:
            ret = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=BASE_DIR)

        status = "OK" if ret.returncode == 0 else "FAIL"
        results.append((tag, seed, status, log_path))
        print(f"[DONE] {tag} -> {status}")

    print("\n" + "=" * 60)
    print("학습 완료 요약")
    print("=" * 60)
    for tag, seed, status, log_path in results:
        print(f"  tag={tag} (seed={seed})  ->  {status}  | log={log_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commander batch train runner")
    parser.add_argument("--count", type=int, default=10, help="학습할 모델 개수")
    parser.add_argument("--leverage", type=int, default=3, help="레버리지 배수")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--improved-hp", action="store_true")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="시드 자동 생성 시작값 (기본 42)")
    parser.add_argument("--seed-step", type=int, default=1,
                        help="시드 자동 생성 간격 (기본 1)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="쉼표 구분 시드 목록. 지정 시 base/step보다 우선")
    parser.add_argument("--model-dir", type=str,
                        default=os.path.join(BASE_DIR, "models", "rl_commander"),
                        help="모델 저장 루트 디렉토리")
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join(BASE_DIR, "logs", "train"),
                        help="학습 로그 저장 디렉토리")
    parser.add_argument("--data-path", type=str,
                        default=os.path.join(BASE_DIR, "data", "base_signals_log.csv"),
                        help="학습/평가에 사용할 입력 데이터 CSV")
    args = parser.parse_args()

    run_train_batch(
        count=args.count,
        leverage=args.leverage,
        timesteps=args.timesteps,
        patience=args.patience,
        improved_hp=args.improved_hp,
        base_seed=args.base_seed,
        seed_step=args.seed_step,
        seeds_csv=args.seeds,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        data_path=args.data_path,
    )
