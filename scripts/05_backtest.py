"""
Commander 일괄 백테스트 실행기 (단일/배치 통합본)

단일 모델 백테스트, 다중 모델 앙상블 투표, 여러 폴더의 일괄 백테스트 및 
베스트 모델 자동 선출까지 이 스크립트 하나로 완벽히 제어합니다.
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO

import warnings
warnings.filterwarnings('ignore')

# ── 경로 설정 (새로운 아키텍처 반영) ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # scripts/
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                       # 프로젝트 루트
SRC_DIR = os.path.join(ROOT_DIR, "src")                      # 소스 코드 디렉토리

# 환경(Env) 및 모델(Models) 임포트를 위해 src 경로 추가
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 폴더 이동 전(현재 구조)에서도 작동하도록 commander 경로 임시 추가
COMMANDER_DIR = os.path.join(ROOT_DIR, "commander")
if COMMANDER_DIR not in sys.path:
    sys.path.insert(0, COMMANDER_DIR)

try:
    from envs.trading_env_baby import BabyLeverageTradingEnv as LeverageTradingEnv
except ImportError:
    from crypto_trading_env_baby import BabyLeverageTradingEnv as LeverageTradingEnv


# ── 유틸리티 함수 ─────────────────────────────────────────────────────────
def _infer_leverage_from_tag(tag):
    match = re.match(r"^lev(\d+)_", str(tag).strip().lower())
    return int(match.group(1)) if match else None

def _resolve_model_path(model_tag, model_dir):
    """태그 이름을 기반으로 정확한 zip 파일 경로를 찾습니다."""
    tag = model_tag[:-4] if model_tag.endswith(".zip") else model_tag
    
    # 1. 태그 이름의 폴더 안의 final_model_ / best_model_ 탐색
    tag_folder = os.path.join(model_dir, tag)
    if os.path.isdir(tag_folder):
        for f in os.listdir(tag_folder):
            if f.endswith(".zip") and ("final" in f or "best" in f):
                return os.path.join(tag_folder, f)
    
    # 2. 태그.zip 파일이 디렉토리에 바로 있는 경우
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

def _majority_vote(actions):
    counts = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
        return 0
    return ranked[0][0]

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
    
    # composite score (육각형 통합 지표)
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
    trades = []
    open_trade = None
    liq_steps = []

    while not done:
        pre_step = env.current_step
        pre_position = env.position
        pre_balance = env.balance
        pre_price = float(env.closes[min(pre_step, env.max_steps - 1)])

        # 앙상블 다수결 투표 로직
        votes = []
        for model in models:
            action, _ = model.predict(obs, deterministic=True)
            act = int(action) if np.ndim(action) == 0 else int(action[0])
            votes.append(act)
        act_val = votes[0] if len(votes) == 1 else _majority_vote(votes)

        obs, reward, terminated, truncated, info = env.step(act_val)
        done = terminated or truncated
        balances.append(env.balance)

        # 청산 감지
        if info.get('liquidated') and pre_position != 0:
            liq_steps.append(pre_step)
            if open_trade is not None:
                net_pct = ((env.balance - open_trade["entry_equity"]) / max(open_trade["entry_equity"], 1e-8)) * 100.0
                trades.append({
                    "side": open_trade["side"],
                    "entry_time": open_trade["entry_time"],
                    "exit_time": _step_to_dt_str(pre_step, all_time_index),
                    "holding_steps": pre_step - open_trade["entry_step"],
                    "net_return_pct": net_pct,
                    "exit_type": "liquidated",
                })
                open_trade = None

        # 진입 기록
        if pre_position == 0 and env.position != 0:
            open_trade = {
                "side": "LONG" if env.position == 1 else "SHORT",
                "margin_size": env.position_size,
                "entry_equity": pre_balance,
                "entry_step": pre_step,
                "entry_time": _step_to_dt_str(pre_step, all_time_index),
                "entry_price": pre_price,
            }

        # 정상 청산 감지
        if pre_position != 0 and env.position == 0 and not info.get('liquidated') and open_trade is not None:
            net_ret = ((env.balance - open_trade["entry_equity"]) / max(open_trade["entry_equity"], 1e-8))
            trades.append({
                "side": open_trade["side"],
                "entry_time": open_trade["entry_time"],
                "exit_time": _step_to_dt_str(pre_step, all_time_index),
                "holding_steps": pre_step - open_trade["entry_step"],
                "net_return_pct": net_ret * 100.0,
                "exit_type": "forced" if done else "manual",
            })
            open_trade = None

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
    plt.figure(figsize=(12, 6))
    plt.plot(all_time_index.iloc[:len(balances)], balances, label=f'{model_tag} ({leverage}x)', color='blue')
    plt.axhline(env.initial_balance, color='red', linestyle='--')
    if liq_steps:
        liq_times = [all_time_index.iloc[s] for s in liq_steps if s < len(all_time_index)]
        liq_bals  = [balances[min(s, len(balances)-1)] for s in liq_steps]
        plt.scatter(liq_times, liq_bals, color='red', s=80, zorder=5, label='Liquidation')
    plt.title(f'Backtest Equity Curve: {model_tag}', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(os.path.join(reports_dir, f"rl_backtest_chart_{model_tag}.png"), dpi=300)
    plt.close()

    # JSON 저장
    with open(os.path.join(reports_dir, f"rl_backtest_summary_{model_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ── 자동 태그 검색기 ────────────────────────────────────────────────────────
def _auto_discover_tags(model_dir):
    """rl_generations 폴더 안에 있는 모든 zip 파일과 훈련 완료 폴더를 찾습니다."""
    if not os.path.isdir(model_dir): return []
    tags = []
    for item in os.listdir(model_dir):
        if item.startswith("."): continue
        # Zip 파일인 경우
        if item.endswith(".zip"):
            tags.append(item[:-4])
        # 폴더인 경우 (내부에 zip이 있는지 확인)
        elif os.path.isdir(os.path.join(model_dir, item)):
            for f in os.listdir(os.path.join(model_dir, item)):
                if f.endswith(".zip"):
                    tags.append(item)
                    break
    return sorted(list(set(tags)))


# ── 배치 매니저 ─────────────────────────────────────────────────────────────
def run_backtest_all(tags, leverage_override, model_dir, data_path, reports_dir, tuning_profile, best_metric):
    results = []
    total = len(tags)
    
    print(f"\\n{'='*60}")
    print(f"🚀 총 {total}개 모델 일괄 백테스트 가동 시작")
    print(f"{'='*60}")

    for idx, tag in enumerate(tags, 1):
        leverage = leverage_override if leverage_override else _infer_leverage_from_tag(tag)
        if leverage is None: leverage = 2  # Fallback

        model_path = _resolve_model_path(tag, model_dir)
        if not model_path:
            print(f"[{idx:02d}/{total}] ❌ 실패: {tag} (모델 파일을 찾을 수 없음)")
            continue
            
        print(f"[{idx:02d}/{total}] ⏳ 백테스트 중: {tag} (Lev: {leverage}x) ... ", end="", flush=True)
        
        try:
            # 단일 백테스트 실행
            summary = run_rl_backtest([model_path], tag, leverage, data_path, reports_dir, tuning_profile)
            results.append({"tag": tag, "leverage": leverage, "summary": summary})
            status = "✅ 청산됨" if summary["liquidated"] else "✅ 완료"
            print(status)
        except Exception as e:
            print(f"❌ 에러 발생 ({str(e)})")
            traceback.print_exc()

    # ── 레버리지별 베스트 모델 산출 ──
    if results:
        grouped = {}
        for r in results:
            grouped.setdefault(r["leverage"], []).append(r)

        best_lines = ["leverage,tag,metric,metric_value,total_return_pct,mdd_pct,sharpe_ratio,liquidated"]
        print(f"\\n{'='*60}")
        print(f"🏆 레버리지별 베스트 모델 (기준: {best_metric})")
        print(f"{'='*60}")
        
        for lev in sorted(grouped.keys()):
            candidates = grouped[lev]
            best = max(candidates, key=lambda x: _metric_value(x["summary"], best_metric))
            s = best["summary"]
            mv = _metric_value(s, best_metric)
            line = (f"{lev},{best['tag']},{best_metric},{mv:.6f},"
                    f"{s.get('total_return_pct', 0.0):.6f},{s.get('mdd_pct', 0.0):.6f},"
                    f"{s.get('sharpe_ratio', 0.0):.6f},{s.get('liquidated', False)}")
            best_lines.append(line)
            
            print(f"[Lev {lev}x] 1등: {best['tag']}")
            print(f"   => 수익률: {s.get('total_return_pct'):+.2f}% | MDD: {s.get('mdd_pct'):.2f}% | 승률: {s.get('win_rate'):.1f}%\\n")

        best_output_path = os.path.join(reports_dir, "best_by_leverage.csv")
        with open(best_output_path, "w", encoding="utf-8") as f:
            f.write("\\n".join(best_lines) + "\\n")
        print(f"💾 베스트 결과 저장 완료: {best_output_path}\\n")


# ── 메인 진입점 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commander 일괄 백테스트 (통합본)")
    
    # 신규 디렉토리 아키텍처 기본값
    default_model_dir = os.path.join(ROOT_DIR, "checkpoints", "rl_generations")
    default_data_path = os.path.join(ROOT_DIR, "data", "signals", "base_signals_log.csv")
    default_reports_dir = os.path.join(ROOT_DIR, "reports")

    parser.add_argument("--tags", type=str, default=None, help="쉼표 구분 태그 (지정 안 하면 폴더 내 전체 자동검색)")
    parser.add_argument("--leverage", type=int, default=None, help="레버리지 강제 지정 (기본: 태그명에서 추론)")
    parser.add_argument("--model-dir", type=str, default=default_model_dir, help="모델 가중치 폴더")
    parser.add_argument("--data-path", type=str, default=default_data_path, help="베이스 신호 데이터 CSV")
    parser.add_argument("--reports-dir", type=str, default=default_reports_dir, help="리포트 및 차트 출력 폴더")
    parser.add_argument("--tuning-profile", type=str, choices=["stable", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--best-metric", type=str, choices=["score", "total_return_pct", "sharpe_ratio", "mdd_pct"], default="score", help="1등 선발 기준")
    
    args = parser.parse_args()

    # 구버전 경로에서 실행했을 때 방어 코드
    if not os.path.exists(args.data_path):
        legacy_path = os.path.join(ROOT_DIR, "data", "commander", "base_signals_log.csv")
        if os.path.exists(legacy_path):
            args.data_path = legacy_path
            
    if not os.path.exists(args.model_dir):
        legacy_model_path = os.path.join(ROOT_DIR, "models", "commander", "candidates")
        if os.path.exists(legacy_model_path):
            args.model_dir = legacy_model_path

    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = _auto_discover_tags(args.model_dir)
        if not tags:
            print(f"[ERROR] '{args.model_dir}' 경로에서 백테스트 가능한 모델을 찾을 수 없습니다.")
            sys.exit(1)

    run_backtest_all(
        tags=tags,
        leverage_override=args.leverage,
        model_dir=args.model_dir,
        data_path=args.data_path,
        reports_dir=args.reports_dir,
        tuning_profile=args.tuning_profile,
        best_metric=args.best_metric
    )