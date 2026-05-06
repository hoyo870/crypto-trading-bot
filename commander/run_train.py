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
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def _next_tag(candidates_dir, leverage, seed):
    os.makedirs(candidates_dir, exist_ok=True)
    prefix = f"lev{int(leverage)}_seed{int(seed)}"
    max_idx = 0
    for f in os.listdir(candidates_dir):
        if not f.endswith(".zip"):
            continue
        stem = f[:-4]
        if not stem.startswith(prefix + "_"):
            continue
        suffix = stem[len(prefix) + 1:]
        if suffix.isdigit() and len(suffix) == 3:
            max_idx = max(max_idx, int(suffix))
    return f"{prefix}_{max_idx + 1:03d}"


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


def _preflight_check(data_path, model_dir):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"입력 데이터 파일이 없습니다: {data_path}")

    df = pd.read_csv(data_path)
    required_cols = {"datetime", "close", "long_score", "short_score", "context_score"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"입력 데이터 필수 컬럼 누락: {missing}")

    if len(df) < 50_000:
        print(f"[WARN] 입력 데이터 행 수가 적습니다: {len(df):,}")

    nan_count = int(df[list(required_cols)].isna().sum().sum())
    if nan_count > 0:
        raise ValueError(f"입력 데이터 필수 컬럼에 NaN이 존재합니다: {nan_count}개")

    os.makedirs(os.path.join(model_dir, "runs"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "candidates"), exist_ok=True)
    print(f"[INFO] preflight 통과 | rows={len(df):,} | model_dir={model_dir}")


def _read_run_score(model_dir, tag):
    eval_npz = os.path.join(model_dir, "runs", tag, "evaluations.npz")
    if not os.path.exists(eval_npz):
        best_model = os.path.join(model_dir, "runs", tag, "best_model.zip")
        return 0.0 if os.path.exists(best_model) else float("-inf")
    try:
        data = np.load(eval_npz)
        results = data.get("results", None)
        if results is None or len(results) == 0:
            return float("-inf")
        return float(np.nanmax(results))
    except Exception:
        return float("-inf")


def _select_top_k_candidates(model_dir, tags, top_k):
    if top_k <= 0:
        return

    scored = [(tag, _read_run_score(model_dir, tag)) for tag in tags]
    scored.sort(key=lambda x: x[1], reverse=True)
    keep = {tag for tag, _ in scored[:min(top_k, len(scored))]}

    candidates_dir = os.path.join(model_dir, "candidates")
    for tag, score in scored:
        p = os.path.join(candidates_dir, f"{tag}.zip")
        if tag not in keep and os.path.exists(p):
            os.remove(p)

    if scored and scored[0][0] in keep:
        best_tag = scored[0][0]
        best_src = os.path.join(candidates_dir, f"{best_tag}.zip")
        if os.path.exists(best_src):
            import shutil
            shutil.copy2(best_src, os.path.join(model_dir, "best_model.zip"))

    print("\n[INFO] top-k 후보 선별 결과")
    for i, (tag, score) in enumerate(scored, start=1):
        mark = "KEEP" if tag in keep else "DROP"
        s = "-inf" if score == float("-inf") else f"{score:.4f}"
        print(f"  {i:02d}. {tag:20s} score={s:>8s} [{mark}]")


def run_train_batch(count, leverage, timesteps, patience, improved_hp,
                    base_seed, seed_step, seeds_csv,
                    model_dir, log_dir, data_path, top_k,
                    split_mode, train_ratio, eval_ratio, train_ep_steps, eval_window,
                    eval_freq):
    os.makedirs(log_dir, exist_ok=True)
    candidates_dir = os.path.join(model_dir, "candidates")
    _preflight_check(data_path=data_path, model_dir=model_dir)

    seeds = _resolve_seeds(
        count=count,
        base_seed=base_seed,
        seed_step=seed_step,
        seeds_csv=seeds_csv,
    )
    print(f"[INFO] 사용 시드 목록({len(seeds)}): {seeds}")

    results = []
    for i in range(count):
        seed = seeds[i]
        tag = _next_tag(candidates_dir, leverage=leverage, seed=seed)
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
            "--split-mode", split_mode,
            "--train-ratio", str(train_ratio),
            "--eval-ratio", str(eval_ratio),
            "--train-ep-steps", str(train_ep_steps),
            "--eval-window", str(eval_window),
            "--eval-freq", str(eval_freq),
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

    _select_top_k_candidates(
        model_dir=model_dir,
        tags=[tag for tag, _, status, _ in results if status == "OK"],
        top_k=top_k,
    )

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
                        default=os.path.join(ROOT_DIR, "models", "commander"),
                        help="모델 저장 루트 디렉토리")
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join(BASE_DIR, "logs", "train"),
                        help="학습 로그 저장 디렉토리")
    parser.add_argument("--data-path", type=str,
                        default=os.path.join(ROOT_DIR, "data", "commander", "base_signals_log.csv"),
                        help="학습/평가에 사용할 입력 데이터 CSV")
    parser.add_argument("--top-k", type=int, default=3,
                        help="학습 완료 후 candidates 유지 개수 (0이면 비활성)")
    parser.add_argument("--split-mode", type=str, choices=["none", "holdout"], default="holdout",
                        help="학습/평가 데이터 분할 모드")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="holdout 모드 학습 비율 (0~1)")
    parser.add_argument("--eval-ratio", type=float, default=0.2,
                        help="holdout 모드 평가 비율 (0~1)")
    parser.add_argument("--train-ep-steps", type=int, default=20_000,
                        help="학습 에피소드 길이")
    parser.add_argument("--eval-window", type=int, default=20_000,
                        help="평가 에피소드 길이")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="EvalCallback 평가 주기 (timesteps 단위)")
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
        top_k=args.top_k,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        train_ep_steps=args.train_ep_steps,
        eval_window=args.eval_window,
        eval_freq=args.eval_freq,
    )
