"""RL Commander 모델 일괄 백테스트 실행기.

- runs/<tag>/best_model.zip 모델을 우선 자동 수집
- runs가 없으면 candidates/<tag>.zip을 fallback으로 수집
- 최신 태그 기준 최대 10개 모델을 순차 백테스트
- 각 백테스트의 전체 stdout/stderr를 파일로 저장
- 터미널에는 핵심 로그만 출력
"""
import os
import re
import sys
import subprocess
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── 백테스트 설정 ───────────────────────────────────────────────
MAX_MODELS = 10
# ────────────────────────────────────────────────────────────────


def _is_block_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.fullmatch(r"-+", stripped) and len(stripped) >= 10:
        return True
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    return False


def _discover_models(runs_dir: str, candidates_dir: str, max_models: int) -> list[tuple[str, str, str]]:
    # 반환: (tag, model_path, source)
    run_models: list[tuple[str, str, str]] = []
    if os.path.isdir(runs_dir):
        for name in os.listdir(runs_dir):
            m = re.fullmatch(r"(m\d{3})", name)
            if not m:
                continue
            tag = m.group(1)
            model_path = os.path.join(runs_dir, name, "best_model.zip")
            if os.path.exists(model_path):
                run_models.append((tag, model_path, "runs"))

    if run_models:
        run_models.sort(key=lambda x: int(x[0][1:]))
        if len(run_models) > max_models:
            run_models = run_models[-max_models:]
        return run_models

    candidate_models: list[tuple[str, str, str]] = []
    if os.path.isdir(candidates_dir):
        for name in os.listdir(candidates_dir):
            m = re.fullmatch(r"(m\d{3})\.zip", name)
            if not m:
                continue
            tag = m.group(1)
            model_path = os.path.join(candidates_dir, name)
            if os.path.exists(model_path):
                candidate_models.append((tag, model_path, "candidates"))

    candidate_models.sort(key=lambda x: int(x[0][1:]))
    if len(candidate_models) > max_models:
        candidate_models = candidate_models[-max_models:]
    return candidate_models


def _run_one(model_tag: str, model_path: str, idx: int, total: int, logs_dir: str) -> tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"backtest_{idx:02d}_{model_tag}_{ts}.log"
    log_path = os.path.join(logs_dir, log_name)

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "backtest_rl_commander.py"),
        "--model-path", model_path,
        "--suffix", model_tag,
    ]

    print(f"\n{'#' * 60}")
    print(f"# 백테스트 {idx}/{total} 시작 (tag={model_tag})")
    print(f"# 로그 파일: {log_path}")
    print(f"{'#' * 60}")

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
    print(f"[DONE] tag={model_tag} -> {status}")
    return status, log_path


def main() -> int:
    model_dir = os.path.join(BASE_DIR, "models", "rl_commander")
    runs_dir = os.path.join(model_dir, "runs")
    candidates_dir = os.path.join(model_dir, "candidates")
    logs_dir = os.path.join(BASE_DIR, "logs", "backtest")
    os.makedirs(logs_dir, exist_ok=True)

    models = _discover_models(runs_dir=runs_dir, candidates_dir=candidates_dir, max_models=MAX_MODELS)

    if not models:
        print("[ERROR] 백테스트할 모델을 찾지 못했습니다.")
        print(f"[HINT] '{runs_dir}/mNNN/best_model.zip' 또는 '{candidates_dir}/mNNN.zip' 파일이 필요합니다.")
        return 1

    source = models[0][2]
    discovered = ", ".join(tag for tag, _, _ in models)
    print(f"[INFO] 발견된 모델({len(models)}개, source={source}): {discovered}")

    results = []
    total = len(models)
    for idx, (tag, model_path, _) in enumerate(models, start=1):
        status, log_path = _run_one(model_tag=tag, model_path=model_path, idx=idx, total=total, logs_dir=logs_dir)
        results.append((tag, status, log_path))

    print(f"\n{'=' * 60}")
    print("백테스트 완료 요약")
    print(f"{'=' * 60}")
    for tag, status, log_path in results:
        print(f"  tag={tag}  ->  {status}  | log={log_path}")
    print(f"{'=' * 60}")

    failed = [r for r in results if not r[1].startswith("OK")]
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
