"""
Commander 일괄 백테스트 실행기

사용법:
    python run_backtest.py                                      # 기본(전체 candidates)
    python run_backtest.py --source runs                        # 전체 runs 기준
    python run_backtest.py --tags lev3_seed42_001               # candidates 태그 지정
    python run_backtest.py --source runs --tags lev3_seed42_001 # runs 태그 지정
"""
import os
import sys
import subprocess
import argparse
import re
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def _infer_leverage_from_tag(tag):
    match = re.match(r"^lev(\d+)_", str(tag).strip().lower())
    if not match:
        return None
    return int(match.group(1))


def _resolve_leverage(tag, leverage_override):
    inferred = _infer_leverage_from_tag(tag)
    if leverage_override is not None:
        if inferred is not None and inferred != leverage_override:
            print(
                f"[WARN] tag={tag} 에 포함된 leverage={inferred}x 와 "
                f"--leverage={leverage_override}x 가 다릅니다. override 값을 사용합니다."
            )
        return leverage_override
    if inferred is None:
        print(
            f"[WARN] tag={tag} 에서 leverage를 추론할 수 없어 기본값 1x 를 사용합니다. "
            "가능하면 lev3_seed42_001 형식의 태그를 사용하세요."
        )
        return 1
    return inferred


def _metric_value(summary, metric):
    if metric == "total_return_pct":
        return float(summary.get("total_return_pct", float("-inf")))
    if metric == "sharpe_ratio":
        return float(summary.get("sharpe_ratio", float("-inf")))
    if metric == "mdd_pct":
        return float(summary.get("mdd_pct", float("-inf")))

    # composite score (default)
    total_return = float(summary.get("total_return_pct", 0.0))
    sharpe = float(summary.get("sharpe_ratio", 0.0))
    mdd = abs(float(summary.get("mdd_pct", 0.0)))
    liquidated = bool(summary.get("liquidated", False))
    liq_penalty = 1000.0 if liquidated else 0.0
    return total_return + sharpe * 10.0 - mdd * 0.5 - liq_penalty


def _run_backtest_one(idx, total, tag, leverage_override, model_dir, log_dir,
                      data_path, reports_dir, source, tuning_profile):
    leverage = _resolve_leverage(tag, leverage_override)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"backtest_{idx:02d}_{tag}_{ts}.log")

    print(f"\n[{idx:02d}/{total}] 백테스트: tag={tag} | leverage={leverage}x ...")
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "backtest_rl_commander.py"),
        "--model-tag", tag,
        "--suffix", tag,
        "--leverage", str(leverage),
        "--model-source", source,
        "--model-dir", model_dir,
        "--data-path", data_path,
        "--reports-dir", reports_dir,
        "--tuning-profile", tuning_profile,
    ]

    with open(log_path, "w") as lf:
        ret = subprocess.run(cmd, capture_output=False,
                             stdout=lf, stderr=subprocess.STDOUT,
                             cwd=BASE_DIR)

    status = "OK" if ret.returncode == 0 else "FAIL"
    summary = None
    if status == "OK":
        summary_path = os.path.join(reports_dir, f"rl_backtest_summary_{tag}.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                summary = None

    return {
        "tag": tag,
        "leverage": leverage,
        "status": status,
        "log_path": log_path,
        "summary": summary,
    }


def _pick_best_per_leverage(results, metric, best_output_path):
    ok_results = [r for r in results if r["status"] == "OK" and r["summary"] is not None]
    grouped = {}
    for r in ok_results:
        lev = int(r["leverage"])
        grouped.setdefault(lev, []).append(r)

    lines = []
    lines.append("leverage,tag,metric,metric_value,total_return_pct,mdd_pct,sharpe_ratio,liquidated")
    for lev in sorted(grouped.keys()):
        candidates = grouped[lev]
        best = max(candidates, key=lambda x: _metric_value(x["summary"], metric))
        s = best["summary"]
        mv = _metric_value(s, metric)
        lines.append(
            f"{lev},{best['tag']},{metric},{mv:.6f},"
            f"{float(s.get('total_return_pct', 0.0)):.6f},"
            f"{float(s.get('mdd_pct', 0.0)):.6f},"
            f"{float(s.get('sharpe_ratio', 0.0)):.6f},"
            f"{bool(s.get('liquidated', False))}"
        )

    out_dir = os.path.dirname(best_output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(best_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[INFO] 레버리지별 베스트 후보")
    for line in lines[1:]:
        print(f"  {line}")
    print(f"[INFO] 베스트 후보 파일 저장: {best_output_path}")


def run_backtest_all(tags, leverage_override, model_dir, log_dir, data_path,
                     reports_dir, source, workers, tuning_profile,
                     pick_best_per_leverage=False, best_metric="score",
                     best_output_path=None):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futures = []
        total = len(tags)
        for i, tag in enumerate(tags, start=1):
            futures.append(
                ex.submit(
                    _run_backtest_one,
                    i, total, tag, leverage_override, model_dir,
                    log_dir, data_path, reports_dir, source, tuning_profile,
                )
            )

        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"[{'DONE' if r['status'] == 'OK' else 'ERROR'}] tag={r['tag']} -> {r['status']}")

    results.sort(key=lambda x: x["tag"])

    print("\n" + "=" * 60)
    print("백테스트 완료 요약")
    print("=" * 60)
    for r in results:
        print(f"  tag={r['tag']}  ->  {r['status']}  | log={r['log_path']}")
    print("=" * 60)

    if pick_best_per_leverage:
        if not best_output_path:
            best_output_path = os.path.join(reports_dir, "best_by_leverage.csv")
        _pick_best_per_leverage(results=results, metric=best_metric, best_output_path=best_output_path)


def _auto_discover_tags(model_dir, source):
    if source == "runs":
        runs_dir = os.path.join(model_dir, "runs")
        if not os.path.isdir(runs_dir):
            return []
        tags = sorted(
            name for name in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, name))
            and os.path.exists(os.path.join(runs_dir, name, "best_model.zip"))
            and not name.startswith(".")
        )
        return tags

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
                        help="쉼표 구분 태그 (source에 따라 candidates 또는 runs에서 조회)")
    parser.add_argument("--source", type=str, choices=["candidates", "runs"], default="candidates",
                        help="백테스트 대상 모델 원본 (기본: candidates)")
    parser.add_argument("--leverage", type=int, default=None,
                        help="레버리지 배수 override (미지정 시 태그에서 자동 추론)")
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
    parser.add_argument("--workers", type=int, default=3,
                        help="병렬 백테스트 스레드 수")
    parser.add_argument("--tuning-profile", type=str,
                        choices=["stable", "balanced", "aggressive"],
                        default="balanced",
                        help="백테스트 환경 튜닝 프로파일")
    parser.add_argument("--pick-best-per-leverage", action="store_true",
                        help="백테스트 완료 후 레버리지별 베스트 후보를 산출")
    parser.add_argument("--best-metric", type=str,
                        choices=["score", "total_return_pct", "sharpe_ratio", "mdd_pct"],
                        default="score",
                        help="베스트 후보 산출 기준")
    parser.add_argument("--best-output", type=str, default=None,
                        help="레버리지별 베스트 후보 출력 CSV 경로")
    args = parser.parse_args()

    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = _auto_discover_tags(args.model_dir, args.source)
        if not tags:
            print(f"[ERROR] {args.source} 에 백테스트 가능한 모델이 없습니다.")
            sys.exit(1)
        print(f"[INFO] 자동 발견된 {args.source} 모델 {len(tags)}개: {tags}")

    run_backtest_all(
        tags=tags,
        leverage_override=args.leverage,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        data_path=args.data_path,
        reports_dir=args.reports_dir,
        source=args.source,
        workers=args.workers,
        tuning_profile=args.tuning_profile,
        pick_best_per_leverage=args.pick_best_per_leverage,
        best_metric=args.best_metric,
        best_output_path=args.best_output,
    )
