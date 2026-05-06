import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from crypto_trading_env import CryptoTradingEnv
import warnings
warnings.filterwarnings('ignore')


def _resolve_model_path(model_path=None, model_tag=None):
    model_dir = os.path.join(BASE_DIR, "models", "rl_commander")
    if model_path is not None:
        if not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)
        return model_path

    if model_tag:
        tag = model_tag[:-4] if model_tag.endswith(".zip") else model_tag
        candidates_path = os.path.join(model_dir, "candidates", f"{tag}.zip")
        legacy_path = os.path.join(model_dir, f"{tag}.zip")
        if os.path.exists(candidates_path):
            return candidates_path
        if os.path.exists(legacy_path):
            return legacy_path
        return candidates_path

    return os.path.join(model_dir, "best_model.zip")


def _resolve_model_paths(model_paths=None, model_tags=None):
    paths = []
    if model_paths:
        for path in model_paths:
            if not os.path.isabs(path):
                path = os.path.join(BASE_DIR, path)
            paths.append(path)
        return paths

    if model_tags:
        for tag in model_tags:
            paths.append(_resolve_model_path(model_tag=tag))
        return paths

    return [_resolve_model_path()]


def _majority_vote(actions):
    # 동률이면 관망(0)으로 보수적 처리
    counts = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
        return 0
    return ranked[0][0]


def _load_all_datetimes(data_path):
    """전체 데이터 기간의 datetime 인덱스 반환 (split 없음)"""
    df_time = pd.read_csv(data_path, usecols=["datetime"])
    return pd.to_datetime(df_time["datetime"], errors="coerce").reset_index(drop=True)


def _step_to_dt_str(step, dt_index):
    if step < 0 or step >= len(dt_index):
        return "N/A"
    value = dt_index.iloc[step]
    if pd.isna(value):
        return "N/A"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _write_trade_report(report_path, model_path, summary, trades):
    lines = []
    lines.append("RL Commander Backtest Detailed Report")
    lines.append("=" * 80)
    lines.append(f"Model Path      : {model_path}")
    lines.append(f"Initial Balance : {summary['initial_balance']:.2f}")
    lines.append(f"Final Balance   : {summary['final_balance']:.2f}")
    lines.append(f"Total Return(%) : {summary['total_return_pct']:+.2f}")
    lines.append(f"Test Period     : {summary['period_start']} ~ {summary['period_end']}")
    lines.append(f"Total Trades    : {summary['total_trades']}")
    lines.append(f"Long Trades     : {summary['long_count']}")
    lines.append(f"Short Trades    : {summary['short_count']}")
    lines.append(f"Win Rate(%)     : {summary['win_rate']:.2f}")
    lines.append("")

    lines.append("Column Guide")
    lines.append("-" * 80)
    lines.append("entry_time     : 포지션 진입 시각 (전체 데이터 기간 datetime 기준)")
    lines.append("exit_time      : 포지션 청산 시각")
    lines.append("entry_step     : 전체 데이터 내 진입 인덱스(5분봉 기준 step)")
    lines.append("exit_step      : 전체 데이터 내 청산 인덱스")
    lines.append("hold           : 보유한 step 수 (1 step = 5분)")
    lines.append("entry_px       : 진입 가격")
    lines.append("exit_px        : 청산 가격")
    lines.append("gross%         : 수수료 제외 수익률(%)")
    lines.append("net%           : 수수료 포함 순수익률(%)")
    lines.append("exit_type      : 청산 유형 (manual=모델 청산, forced=에피소드 종료 강제청산)")
    lines.append("")

    lines.append("Trade Logs")
    lines.append("-" * 80)
    if not trades:
        lines.append("No trades executed.")
    else:
        header = (
            f"{'#':>4} | {'side':>5} | {'entry_time':>19} | {'exit_time':>19} | "
            f"{'entry_step':>10} | {'exit_step':>9} | "
            f"{'hold':>4} | {'entry_px':>12} | {'exit_px':>12} | "
            f"{'gross%':>8} | {'net%':>8} | {'exit_type':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for i, t in enumerate(trades, start=1):
            lines.append(
                f"{i:>4} | {t['side']:>5} | {t['entry_time']:>19} | {t['exit_time']:>19} | "
                f"{t['entry_step']:>10} | {t['exit_step']:>9} | "
                f"{t['holding_steps']:>4} | {t['entry_price']:>12.4f} | {t['exit_price']:>12.4f} | "
                f"{t['gross_return_pct']:>8.3f} | {t['net_return_pct']:>8.3f} | {t['exit_type']:>10}"
            )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_rl_backtest(model_path=None, model_tag=None, output_suffix="", ensemble_tags=None):
    print(f"\n{'='*50}")
    print(f"📈 3차 사령관 (RL Commander) 실전 백테스트 시작")
    print(f"{'='*50}")

    if ensemble_tags:
        tags = [t.strip() for t in ensemble_tags.split(",") if t.strip()]
        model_paths = _resolve_model_paths(model_tags=tags)
    else:
        model_paths = _resolve_model_paths(model_paths=[model_path] if model_path else None,
                                           model_tags=[model_tag] if model_tag else None)

    data_path = os.path.join(BASE_DIR, "data", "base_signals_log.csv")
    all_time_index = _load_all_datetimes(data_path)

    for path in model_paths:
        if not os.path.exists(path):
            print(f"[ERROR] 모델 파일이 없습니다: {path}")
            return

    # 1. 평가용 환경 로드 (전체 데이터 기간, start_step=0 고정)
    env = CryptoTradingEnv(data_path=data_path)

    # 2. 훈련된 사령관 로드
    print("[INFO] 최고 성능의 사령관 뇌를 이식 중...")
    models = [PPO.load(path) for path in model_paths]
    if len(models) > 1:
        print(f"[INFO] 앙상블 모드: {len(models)}개 모델 다수결")

    # 백테스트는 항상 step 0부터 전체 데이터를 순서대로 평가
    obs, _ = env.reset(options={'start_step': 0, 'max_ep_steps': None})

    done = False
    
    # 기록용 리스트
    balances = [env.balance]
    entry_actions = []
    trades = []
    open_trade = None

    print("[INFO] 백테스트 시뮬레이션 가동 중...")
    
    while not done:
        pre_step = env.current_step
        pre_position = env.position
        pre_price = float(env.closes[pre_step])

        # 결정론적 행동 선택
        votes = []
        for model in models:
            action, _states = model.predict(obs, deterministic=True)
            act = int(action) if np.ndim(action) == 0 else int(action[0])
            votes.append(act)
        act_val = votes[0] if len(votes) == 1 else _majority_vote(votes)
        can_enter = (env.position == 0)
        
        # Gymnasium 표준 5-tuple 반환
        obs, reward, terminated, truncated, info = env.step(act_val)
        done = terminated or truncated
        
        balances.append(env.balance)

        # 실제 진입이 가능한 상태에서 발생한 진입 행동만 기록
        if can_enter and act_val in [1, 2]:
            entry_actions.append(act_val)

        # 진입 기록
        if pre_position == 0 and act_val in [1, 2]:
            open_trade = {
                "side": "LONG" if act_val == 1 else "SHORT",
                "entry_step": pre_step,
                "entry_time": _step_to_dt_str(pre_step, all_time_index),
                "entry_price": pre_price,
            }

        # 수동 청산 기록
        if pre_position != 0 and act_val == 3 and open_trade is not None:
            if pre_position == 1:
                gross_ret = (pre_price - open_trade["entry_price"]) / open_trade["entry_price"]
            else:
                gross_ret = (open_trade["entry_price"] - pre_price) / open_trade["entry_price"]
            net_ret = gross_ret - (env.fee_rate * 2)
            trades.append({
                "side": open_trade["side"],
                "entry_time": open_trade["entry_time"],
                "exit_time": _step_to_dt_str(pre_step, all_time_index),
                "entry_step": open_trade["entry_step"],
                "exit_step": pre_step,
                "holding_steps": pre_step - open_trade["entry_step"],
                "entry_price": open_trade["entry_price"],
                "exit_price": pre_price,
                "gross_return_pct": gross_ret * 100.0,
                "net_return_pct": net_ret * 100.0,
                "exit_type": "manual",
            })
            open_trade = None

        # 에피소드 종료 강제청산 기록
        if done and pre_position != 0 and act_val != 3 and open_trade is not None:
            forced_exit_step = min(env.current_step, env.max_steps - 1)
            forced_exit_price = float(env.closes[forced_exit_step])
            if pre_position == 1:
                gross_ret = (forced_exit_price - open_trade["entry_price"]) / open_trade["entry_price"]
            else:
                gross_ret = (open_trade["entry_price"] - forced_exit_price) / open_trade["entry_price"]
            net_ret = gross_ret - (env.fee_rate * 2)
            trades.append({
                "side": open_trade["side"],
                "entry_time": open_trade["entry_time"],
                "exit_time": _step_to_dt_str(forced_exit_step, all_time_index),
                "entry_step": open_trade["entry_step"],
                "exit_step": forced_exit_step,
                "holding_steps": forced_exit_step - open_trade["entry_step"],
                "entry_price": open_trade["entry_price"],
                "exit_price": forced_exit_price,
                "gross_return_pct": gross_ret * 100.0,
                "net_return_pct": net_ret * 100.0,
                "exit_type": "forced",
            })
            open_trade = None

    # 3. 결과 요약
    final_balance = info['final_balance']
    pnl_pct = ((final_balance - env.initial_balance) / env.initial_balance) * 100
    total_trades = info['total_trades']
    win_rate = info.get('win_rate', 0.0)

    long_count = entry_actions.count(1)
    short_count = entry_actions.count(2)

    print("\n========================================")
    print("📊 강화학습 사령관 백테스트 결과 리포트")
    print("========================================")
    print(f"초기 자본금 : {env.initial_balance:,.2f} USDT")
    print(f"최종 자본금 : {final_balance:,.2f} USDT")
    print(f"총 수익률   : {pnl_pct:>+8.2f}%")
    print(f"총 거래횟수 : {total_trades}회 (Long: {long_count} / Short: {short_count})")
    print(f"승률        : {win_rate:.2f}%")
    print("========================================")

    evaluated_end_step = min(env.current_step, env.max_steps - 1)

    summary = {
        "initial_balance": env.initial_balance,
        "final_balance": final_balance,
        "total_return_pct": pnl_pct,
        "period_start": _step_to_dt_str(0, all_time_index),
        "period_end": _step_to_dt_str(evaluated_end_step, all_time_index),
        "total_trades": total_trades,
        "long_count": long_count,
        "short_count": short_count,
        "win_rate": win_rate,
    }

    # 4. 자산 곡선(Equity Curve) 시각화 및 저장
    x_times = all_time_index.iloc[:len(balances)]

    plt.figure(figsize=(12, 6))
    plt.plot(x_times, balances, label='RL Commander Balance', color='blue', linewidth=1.5)
    plt.axhline(env.initial_balance, color='red', linestyle='--', label='Initial Balance')
    
    plt.title('RL Commander Equity Curve (Full Data: 2023-05-06 ~ 2026-05-04)', fontsize=14, fontweight='bold')
    plt.xlabel('Datetime', fontsize=12)
    plt.ylabel('Balance (USDT)', fontsize=12)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if not output_suffix and model_tag:
        output_suffix = model_tag

    # 결과 파일은 reports/ 폴더에 통합 관리
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    if output_suffix:
        save_name = f"rl_backtest_result_{output_suffix}.png"
        report_name = f"rl_backtest_report_{output_suffix}.txt"
    else:
        save_name = "rl_backtest_result.png"
        report_name = "rl_backtest_report.txt"

    save_fig_path = os.path.join(reports_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    print(f"\n[INFO] 📈 결과 차트가 '{save_fig_path}' 파일로 저장되었습니다!")

    report_path = os.path.join(reports_dir, report_name)
    model_desc = ", ".join(model_paths) if len(model_paths) > 1 else model_paths[0]
    _write_trade_report(report_path=report_path, model_path=model_desc, summary=summary, trades=trades)
    print(f"[INFO] 📝 상세 리포트가 '{report_path}' 파일로 저장되었습니다!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest RL commander with selected model")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Absolute path or v2-relative path of RL model zip")
    parser.add_argument("--model-tag", type=str, default=None,
                        help="Short tag in models/rl_commander/candidates, e.g. m001")
    parser.add_argument("--ensemble-tags", type=str, default=None,
                        help="Comma-separated tags for ensemble voting, e.g. m010,m002")
    parser.add_argument("--suffix", type=str, default="",
                        help="Output image suffix. e.g. seed42")
    args = parser.parse_args()

    run_rl_backtest(model_path=args.model_path,
                    model_tag=args.model_tag,
                    output_suffix=args.suffix,
                    ensemble_tags=args.ensemble_tags)