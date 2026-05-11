"""
Commander 일괄 백테스트 실행기 (병렬 멀티프로세싱 최적화)

단일 모델 백테스트, 다중 모델 앙상블 투표, 여러 폴더의 일괄 백테스트를
M1 Max 멀티 코어를 활용하여 초고속으로 동시 처리합니다.
"""

import os
import sys
import argparse
import json
import re
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import logging

# 멀티프로세싱 중 차트 충돌(GUI 에러) 방지를 위한 Agg 백엔드 강제
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from stable_baselines3 import PPO
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ── 경로 설정 ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                       
SRC_DIR = os.path.join(ROOT_DIR, "src")                      

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.envs.trading_env_baby import BabyLeverageTradingEnv as LeverageTradingEnv

# ── 로깅 설정 ─────────────────────────────────────────────────────────────
os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "orchestrator.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Backtest")

# ── 세대별 로그 파일 추가 (CUSTOM_LOG_DIR 환경변수 세팅 시 backtest.log 병행 기록) ──────
_custom_log_dir = os.environ.get("CUSTOM_LOG_DIR")
if _custom_log_dir:
    os.makedirs(_custom_log_dir, exist_ok=True)
    logging.getLogger().addHandler(
        logging.FileHandler(os.path.join(_custom_log_dir, "backtest.log"), encoding='utf-8')
    )


# ── 유틸리티 함수 ─────────────────────────────────────────────────────────
def _infer_leverage_from_tag(tag):
    match = re.match(r"^lev(\d+)_", str(tag).strip().lower())
    return int(match.group(1)) if match else None

def _resolve_model_path(model_tag, model_dir):
    tag = model_tag[:-4] if model_tag.endswith(".zip") else model_tag
    tag_folder = os.path.join(model_dir, tag)
    if os.path.isdir(tag_folder):
        for f in os.listdir(tag_folder):
            if f.endswith(".zip") and ("final" in f or "best" in f):
                return os.path.join(tag_folder, f)
    direct_zip = os.path.join(model_dir, f"{tag}.zip")
    if os.path.exists(direct_zip):
        return direct_zip
    return None

def _calc_mdd(balances):
    arr = np.array(balances, dtype=float)
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak
    return float(drawdown.min()) * 100.0

def _calc_sharpe(balances, steps_per_year=105120):
    arr = np.array(balances, dtype=float)
    if len(arr) < 2: return 0.0
    rets = np.diff(arr) / arr[:-1]
    mu, sigma = rets.mean(), rets.std(ddof=1)
    if sigma == 0: return 0.0
    return float(mu / sigma * np.sqrt(steps_per_year))

def _load_all_datetimes(data_path):
    df = pd.read_csv(data_path, usecols=["datetime"])
    return pd.to_datetime(df["datetime"], errors="coerce").reset_index(drop=True)

def _step_to_dt_str(step, dt_index):
    if step < 0 or step >= len(dt_index): return "N/A"
    val = dt_index.iloc[step]
    return "N/A" if pd.isna(val) else val.strftime("%Y-%m-%d %H:%M:%S")

def _metric_value(summary, metric):
    if metric == "total_return_pct": return float(summary.get("total_return_pct", float("-inf")))
    if metric == "sharpe_ratio": return float(summary.get("sharpe_ratio", float("-inf")))
    if metric == "mdd_pct": return float(summary.get("mdd_pct", float("-inf")))
    
    total_return = float(summary.get("total_return_pct", 0.0))
    sharpe = float(summary.get("sharpe_ratio", 0.0))
    mdd = abs(float(summary.get("mdd_pct", 0.0)))
    liquidated = bool(summary.get("liquidated", False))
    liq_penalty = 1000.0 if liquidated else 0.0
    return total_return + sharpe * 10.0 - mdd * 0.5 - liq_penalty


# ── 코어 백테스트 실행 엔진 ──────────────────────────────────────────────────
def run_rl_backtest(model_paths, model_tag, leverage, data_path, reports_dir, tuning_profile="balanced"):
    all_time_index = _load_all_datetimes(data_path)
    env = LeverageTradingEnv(data_path=data_path, leverage=leverage, tuning_profile=tuning_profile)

    models = [PPO.load(path) for path in model_paths]
    obs, _ = env.reset(options={'start_step': 0, 'max_ep_steps': None})

    done = False
    balances = [env.balance]
    liq_steps = []

    # 병렬 처리 중 터미널 로그 꼬임 방지를 위해 print문 최소화
    while not done:
        pre_step = env.current_step
        pre_position = env.position
        
        votes = []
        for model in models:
            action, _ = model.predict(obs, deterministic=True)
            act = int(action) if np.ndim(action) == 0 else int(action[0])
            votes.append(act)
        
        # 단순 다수결
        if len(votes) == 1: act_val = votes[0]
        else:
            counts = {a: votes.count(a) for a in set(votes)}
            act_val = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

        obs, reward, terminated, truncated, info = env.step(act_val)
        done = terminated or truncated
        balances.append(env.balance)

        if info.get('liquidated') and pre_position != 0:
            liq_steps.append(pre_step)

    # 요약 정보 산출
    final_balance = info.get('final_balance', env.balance)
    pnl_pct = ((final_balance - env.initial_balance) / env.initial_balance) * 100
    mdd = _calc_mdd(balances)
    sharpe = _calc_sharpe(balances)

    summary = {
        "model_tag": model_tag,
        "initial_balance": env.initial_balance,
        "final_balance": final_balance,
        "total_return_pct": pnl_pct,
        "mdd_pct": mdd,
        "sharpe_ratio": sharpe,
        "total_trades": info.get('total_trades', 0),
        "win_rate": info.get('win_rate', 0.0),
        "leverage": leverage,
        "liquidated": info.get('liquidated', False),
    }

    # 차트 그리기 및 저장
    plt.figure(figsize=(10, 5))
    plt.plot(all_time_index.iloc[:len(balances)], balances, label=f'{model_tag} ({leverage}x)', color='blue')
    plt.axhline(env.initial_balance, color='red', linestyle='--')
    if liq_steps:
        liq_times = [all_time_index.iloc[s] for s in liq_steps if s < len(all_time_index)]
        liq_bals  = [balances[min(s, len(balances)-1)] for s in liq_steps]
        plt.scatter(liq_times, liq_bals, color='red', s=60, zorder=5, label='Liquidation')
    plt.title(f'Backtest: {model_tag}', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(os.path.join(reports_dir, f"rl_backtest_chart_{model_tag}.png"), dpi=200)
    plt.close()

    with open(os.path.join(reports_dir, f"rl_backtest_summary_{model_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ── 멀티프로세싱 워커(Worker) 함수 ─────────────────────────────────────────
def _backtest_worker(tag, leverage_override, model_dir, data_path, reports_dir, tuning_profile):
    try:
        leverage = leverage_override if leverage_override else _infer_leverage_from_tag(tag)
        if leverage is None: leverage = 2 

        model_path = _resolve_model_path(tag, model_dir)
        if not model_path:
            return tag, leverage, None, "모델 파일을 찾을 수 없음"
            
        summary = run_rl_backtest([model_path], tag, leverage, data_path, reports_dir, tuning_profile)
        return tag, leverage, summary, None
    except Exception as e:
        return tag, leverage_override, None, str(e)


# ── 병렬 배치 매니저 ────────────────────────────────────────────────────────
def run_backtest_all(tags, leverage_override, model_dir, data_path, reports_dir, tuning_profile, best_metric, jobs):
    results = []
    total = len(tags)
    
    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"🚀 총 {total}개 모델 병렬 일괄 백테스트 가동 시작 (코어: {jobs}개)")
    logger.info(f"{'='*70}")

    # M1 Max 코어를 100% 활용하는 ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_backtest_worker, tag, leverage_override, model_dir, data_path, reports_dir, tuning_profile): tag 
            for tag in tags
        }
        
        completed = 0
        for future in as_completed(futures):
            tag, lev, summary, err = future.result()
            completed += 1
            
            if err:
                logger.error(f"[{completed:03d}/{total}] ❌ 실패: {tag} ({err})")
            else:
                status = "⚠️ 청산" if summary["liquidated"] else "✅ 완료"
                logger.info(f"[{completed:03d}/{total}] {status}: {tag} (Lev: {lev}x) => 수익률: {summary.get('total_return_pct', 0):+7.2f}% | MDD: {summary.get('mdd_pct', 0):5.2f}%")
                results.append({"tag": tag, "leverage": lev, "summary": summary})

    # ── 레버리지별 베스트 모델 산출 ──
    if results:
        grouped = {}
        for r in results:
            grouped.setdefault(r["leverage"], []).append(r)

        best_lines = ["leverage,tag,metric,metric_value,total_return_pct,mdd_pct,sharpe_ratio,liquidated"]
        logger.info("")
        logger.info(f"{'='*70}")
        logger.info(f"🏆 레버리지별 베스트 모델 (기준: {best_metric})")
        logger.info(f"{'='*70}")
        
        for lev in sorted(grouped.keys()):
            candidates = grouped[lev]
            best = max(candidates, key=lambda x: _metric_value(x["summary"], best_metric))
            s = best["summary"]
            mv = _metric_value(s, best_metric)
            line = (f"{lev},{best['tag']},{best_metric},{mv:.6f},"
                    f"{s.get('total_return_pct', 0.0):.6f},{s.get('mdd_pct', 0.0):.6f},"
                    f"{s.get('sharpe_ratio', 0.0):.6f},{s.get('liquidated', False)}")
            best_lines.append(line)
            
            logger.info(f"[Lev {lev}x] 1등: {best['tag']}")
            logger.info(f"   => 수익률: {s.get('total_return_pct'):+.2f}% | MDD: {s.get('mdd_pct'):.2f}% | 승률: {s.get('win_rate'):.1f}%\n")

        best_output_path = os.path.join(reports_dir, "best_by_leverage.csv")
        with open(best_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(best_lines) + "\n")
        logger.info(f"💾 베스트 결과 저장 완료: {best_output_path}\n")


def _auto_discover_tags(model_dir):
    if not os.path.isdir(model_dir): return []
    tags = []
    for item in os.listdir(model_dir):
        if item.startswith("."): continue
        if item.endswith(".zip"): tags.append(item[:-4])
        elif os.path.isdir(os.path.join(model_dir, item)):
            for f in os.listdir(os.path.join(model_dir, item)):
                if f.endswith(".zip"):
                    tags.append(item)
                    break
    return sorted(list(set(tags)))


# ── 메인 진입점 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commander 일괄 백테스트 (병렬 지원)")
    
    default_model_dir = os.path.join(ROOT_DIR, "checkpoints", "rl_generations")
    default_data_path = os.path.join(ROOT_DIR, "data", "signals", "base_signals_log.csv")
    default_reports_dir = os.path.join(ROOT_DIR, "reports")

    parser.add_argument("--tags", type=str, default=None, help="쉼표 구분 태그 (지정 안 하면 폴더 내 전체 스캔)")
    parser.add_argument("--leverage", type=int, default=None, help="레버리지 강제 지정")
    parser.add_argument("--model-dir", type=str, default=default_model_dir, help="모델 가중치 폴더")
    parser.add_argument("--data-path", type=str, default=default_data_path, help="베이스 신호 데이터 CSV")
    parser.add_argument("--reports-dir", type=str, default=default_reports_dir, help="리포트 및 차트 출력 폴더")
    parser.add_argument("--tuning-profile", type=str, choices=["stable", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--best-metric", type=str, choices=["score", "total_return_pct", "sharpe_ratio", "mdd_pct"], default="score")
    parser.add_argument("--jobs", type=int, default=5, help="백테스트 동시 실행 프로세스 수 (기본: 5)")
    
    args = parser.parse_args()

    # 환경변수 연동 (run_evolution.py 호환성)
    args.model_dir = os.environ.get("CUSTOM_MODEL_DIR", args.model_dir)

    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = _auto_discover_tags(args.model_dir)
        if not tags:
            logger.error(f"[ERROR] '{args.model_dir}' 경로에서 백테스트 가능한 모델을 찾을 수 없습니다.")
            sys.exit(1)

    run_backtest_all(
        tags=tags,
        leverage_override=args.leverage,
        model_dir=args.model_dir,
        data_path=args.data_path,
        reports_dir=args.reports_dir,
        tuning_profile=args.tuning_profile,
        best_metric=args.best_metric,
        jobs=args.jobs
    )