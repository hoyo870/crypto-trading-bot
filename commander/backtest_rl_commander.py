"""Commander RL 백테스트 스크립트 (통합 레버리지 환경)."""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from crypto_trading_env import LeverageTradingEnv
import warnings
warnings.filterwarnings('ignore')


# ── 유틸 ─────────────────────────────────────────────────────────
def _resolve_model_path(model_path=None, model_tag=None, model_dir=None, model_source="candidates"):
    if model_dir is None:
        model_dir = os.path.join(ROOT_DIR, "models", "commander")
    if model_path is not None:
        if not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)
        return model_path
    if model_tag:
        tag = model_tag[:-4] if model_tag.endswith(".zip") else model_tag
        if model_source == "runs":
            run_model_path = os.path.join(model_dir, "runs", tag, "best_model.zip")
            if os.path.exists(run_model_path):
                return run_model_path
            return run_model_path
        candidates_path = os.path.join(model_dir, "candidates", f"{tag}.zip")
        if os.path.exists(candidates_path):
            return candidates_path
        return candidates_path
    return os.path.join(model_dir, "best_model.zip")


def _resolve_model_paths(model_paths=None, model_tags=None, model_dir=None, model_source="candidates"):
    if model_paths:
        return [p if os.path.isabs(p) else os.path.join(BASE_DIR, p) for p in model_paths]
    if model_tags:
        return [_resolve_model_path(model_tag=t, model_dir=model_dir, model_source=model_source) for t in model_tags]
    return [_resolve_model_path(model_dir=model_dir, model_source=model_source)]


def _calc_mdd(balances):
    arr = np.array(balances, dtype=float)
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak
    return float(drawdown.min()) * 100.0


def _calc_sharpe(balances, steps_per_year=105120):
    arr = np.array(balances, dtype=float)
    if len(arr) < 2:
        return 0.0
    rets = np.diff(arr) / arr[:-1]
    mu, sigma = rets.mean(), rets.std(ddof=1)
    if sigma == 0:
        return 0.0
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
    if step < 0 or step >= len(dt_index):
        return "N/A"
    val = dt_index.iloc[step]
    return "N/A" if pd.isna(val) else val.strftime("%Y-%m-%d %H:%M:%S")


# ── 리포트 작성 ──────────────────────────────────────────────────
def _write_trade_report(report_path, model_path, summary, trades):
    lines = []
    lines.append("Commander RL Backtest Detailed Report (Leverage)")
    lines.append("=" * 80)
    lines.append(f"Model Path      : {model_path}")
    lines.append(f"Leverage        : {summary['leverage']}x")
    lines.append(f"Initial Balance : {summary['initial_balance']:.2f} USDT")
    lines.append(f"Final Balance   : {summary['final_balance']:.2f} USDT")
    lines.append(f"Total Return(%) : {summary['total_return_pct']:+.2f}")
    lines.append(f"Max Drawdown(%) : {summary['mdd_pct']:+.2f}")
    lines.append(f"Sharpe Ratio    : {summary['sharpe_ratio']:.4f}")
    lines.append(f"Test Period     : {summary['period_start']} ~ {summary['period_end']}")
    lines.append(f"Total Trades    : {summary['total_trades']}")
    lines.append(f"Long Trades     : {summary['long_count']}")
    lines.append(f"Short Trades    : {summary['short_count']}")
    lines.append(f"Win Rate(%)     : {summary['win_rate']:.2f}")
    lines.append(f"Liquidated      : {summary['liquidated']}")
    lines.append("")

    lines.append("Trade Logs")
    lines.append("-" * 80)
    if not trades:
        lines.append("No trades executed.")
    else:
        header = (
            f"{'#':>4} | {'side':>5} | {'entry_time':>19} | {'exit_time':>19} | "
            f"{'hold':>4} | {'entry_px':>12} | {'exit_px':>12} | "
            f"{'gross%':>8} | {'net%':>8} | {'exit_type':>12}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for i, t in enumerate(trades, start=1):
            lines.append(
                f"{i:>4} | {t['side']:>5} | {t['entry_time']:>19} | {t['exit_time']:>19} | "
                f"{t['holding_steps']:>4} | {t['entry_price']:>12.4f} | {t['exit_price']:>12.4f} | "
                f"{t['gross_return_pct']:>8.3f} | {t['net_return_pct']:>8.3f} | {t['exit_type']:>12}"
            )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── 메인 백테스트 ─────────────────────────────────────────────────
def run_rl_backtest(model_path=None, model_tag=None, output_suffix="",
                    ensemble_tags=None, leverage=2,
                    model_source="candidates",
                    model_dir=None, data_path=None, reports_dir=None):

    print(f"\n{'='*55}")
    print(f"📈 Commander RL 백테스트 시작 (레버리지 {leverage}x)")
    print(f"{'='*55}")

    if ensemble_tags:
        tags = [t.strip() for t in ensemble_tags.split(",") if t.strip()]
        model_paths = _resolve_model_paths(model_tags=tags, model_dir=model_dir, model_source=model_source)
    else:
        model_paths = _resolve_model_paths(
            model_paths=[model_path] if model_path else None,
            model_tags=[model_tag] if model_tag else None,
            model_dir=model_dir,
            model_source=model_source,
        )

    if data_path is None:
        data_path = os.path.join(ROOT_DIR, "data", "commander", "base_signals_log.csv")
    all_time_index = _load_all_datetimes(data_path)

    for path in model_paths:
        if not os.path.exists(path):
            print(f"[ERROR] 모델 파일 없음: {path}")
            return

    # 레버리지 환경 생성 (전체 기간 백테스트)
    env = LeverageTradingEnv(data_path=data_path, leverage=leverage)

    print("[INFO] 모델 로드 중...")
    models = [PPO.load(path) for path in model_paths]
    if len(models) > 1:
        print(f"[INFO] 앙상블 모드: {len(models)}개 모델 다수결")

    obs, _ = env.reset(options={'start_step': 0, 'max_ep_steps': None})

    done = False
    balances = [env.balance]
    trades = []
    open_trade = None
    liq_steps = []  # 청산 발생 step (차트 표시용)

    print("[INFO] 백테스트 시뮬레이션 가동 중...")

    while not done:
        pre_step = env.current_step
        pre_position = env.position
        pre_price = float(env.closes[min(pre_step, env.max_steps - 1)])

        votes = []
        for model in models:
            action, _ = model.predict(obs, deterministic=True)
            act = int(action) if np.ndim(action) == 0 else int(action[0])
            votes.append(act)
        act_val = votes[0] if len(votes) == 1 else _majority_vote(votes)

        obs, reward, terminated, truncated, info = env.step(act_val)
        done = terminated or truncated

        balances.append(env.balance)

        # ── 청산(Liquidation) 감지 ──────────────────────────────
        if info.get('liquidated') and pre_position != 0:
            liq_steps.append(pre_step)
            if open_trade is not None:
                gross_pct = -100.0 / leverage
                trades.append({
                    "side": open_trade["side"],
                    "entry_time": open_trade["entry_time"],
                    "exit_time": _step_to_dt_str(pre_step, all_time_index),
                    "entry_step": open_trade["entry_step"],
                    "exit_step": pre_step,
                    "holding_steps": pre_step - open_trade["entry_step"],
                    "entry_price": open_trade["entry_price"],
                    "exit_price": pre_price,
                    "gross_return_pct": gross_pct,
                    "net_return_pct": gross_pct,
                    "exit_type": "liquidated",
                })
                open_trade = None

        # ── 진입 기록 (v4: 1=long_full, 2=long_half, 3=short_full, 4=short_half) ──
        if pre_position == 0 and env.position != 0:
            open_trade = {
                "side": "LONG" if env.position == 1 else "SHORT",
                "margin_size": env.position_size,
                "entry_step": pre_step,
                "entry_time": _step_to_dt_str(pre_step, all_time_index),
                "entry_price": pre_price,
            }

        # ── 포지션 청산 감지 (manual / stop_loss / forced) ────────
        if pre_position != 0 and env.position == 0 \
                and not info.get('liquidated') and open_trade is not None:
            if pre_position == 1:
                gross_ret = (pre_price - open_trade["entry_price"]) / open_trade["entry_price"] * leverage
            else:
                gross_ret = (open_trade["entry_price"] - pre_price) / open_trade["entry_price"] * leverage
            net_ret = gross_ret - (env.fee_rate * leverage * 2)

            if info.get('stop_loss'):
                exit_type = "stop_loss"
            elif done:
                exit_type = "forced"
            else:
                exit_type = "manual"

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
                "exit_type": exit_type,
            })
            open_trade = None

        # ── 에피소드 종료 시 미청산 포지션 강제 정리 ──────────────
        if done and open_trade is not None and not info.get('liquidated'):
            forced_step = min(env.current_step, env.max_steps - 1)
            forced_price = float(env.closes[forced_step])
            if pre_position == 1:
                gross_ret = (forced_price - open_trade["entry_price"]) / open_trade["entry_price"] * leverage
            else:
                gross_ret = (open_trade["entry_price"] - forced_price) / open_trade["entry_price"] * leverage
            net_ret = gross_ret - (env.fee_rate * leverage * 2)
            trades.append({
                "side": open_trade["side"],
                "entry_time": open_trade["entry_time"],
                "exit_time": _step_to_dt_str(forced_step, all_time_index),
                "entry_step": open_trade["entry_step"],
                "exit_step": forced_step,
                "holding_steps": forced_step - open_trade["entry_step"],
                "entry_price": open_trade["entry_price"],
                "exit_price": forced_price,
                "gross_return_pct": gross_ret * 100.0,
                "net_return_pct": net_ret * 100.0,
                "exit_type": "forced",
            })
            open_trade = None

    # ── 결과 요약 ────────────────────────────────────────────────
    final_balance = info.get('final_balance', env.balance)
    pnl_pct = ((final_balance - env.initial_balance) / env.initial_balance) * 100
    total_trades = info.get('total_trades', 0)
    win_rate = info.get('win_rate', 0.0)
    was_liquidated = info.get('liquidated', False)

    long_count  = sum(1 for t in trades if t['side'] == 'LONG')
    short_count = sum(1 for t in trades if t['side'] == 'SHORT')
    mdd = _calc_mdd(balances)
    sharpe = _calc_sharpe(balances)

    print("\n" + "=" * 55)
    print(f"📊 Commander RL 백테스트 결과 (레버리지 {leverage}x)")
    print("=" * 55)
    print(f"초기 자본금 : {env.initial_balance:,.2f} USDT")
    print(f"최종 자본금 : {final_balance:,.2f} USDT")
    print(f"총 수익률   : {pnl_pct:>+8.2f}%")
    print(f"최대 낙폭   : {mdd:>+8.2f}%")
    print(f"샤프 비율   : {sharpe:>8.4f}")
    print(f"총 거래횟수 : {total_trades}회 (Long: {long_count} / Short: {short_count})")
    print(f"승률        : {win_rate:.2f}%")
    print(f"청산(Liq)   : {'⚠️  발생' if was_liquidated else '없음'}")
    print("=" * 55)

    evaluated_end_step = min(env.current_step, env.max_steps - 1)
    summary = {
        "initial_balance": env.initial_balance,
        "final_balance": final_balance,
        "total_return_pct": pnl_pct,
        "mdd_pct": mdd,
        "sharpe_ratio": sharpe,
        "period_start": _step_to_dt_str(0, all_time_index),
        "period_end": _step_to_dt_str(evaluated_end_step, all_time_index),
        "total_trades": total_trades,
        "long_count": long_count,
        "short_count": short_count,
        "win_rate": win_rate,
        "leverage": leverage,
        "liquidated": was_liquidated,
    }

    # ── 차트 ────────────────────────────────────────────────────
    x_times = all_time_index.iloc[:len(balances)]

    plt.figure(figsize=(12, 6))
    plt.plot(x_times, balances, label=f'RL Commander {leverage}x', color='blue', linewidth=1.5)
    plt.axhline(env.initial_balance, color='red', linestyle='--', label='Initial Balance')

    # 청산 이벤트 빨간 점 표시
    if liq_steps:
        liq_times = [all_time_index.iloc[s] for s in liq_steps if s < len(all_time_index)]
        liq_bals  = [balances[min(s, len(balances)-1)] for s in liq_steps]
        plt.scatter(liq_times, liq_bals, color='red', s=80, zorder=5, label='Liquidation')

    plt.title(f'Commander RL Equity Curve ({leverage}x Leverage)\n'
              f'2023-05-06 ~ 2026-05-04', fontsize=13, fontweight='bold')
    plt.xlabel('Datetime', fontsize=11)
    plt.ylabel('Balance (USDT)', fontsize=11)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if not output_suffix and model_tag:
        output_suffix = model_tag

    if reports_dir is None:
        reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    suffix = output_suffix or "latest"
    save_fig_path = os.path.join(reports_dir, f"rl_backtest_result_{suffix}.png")
    report_path   = os.path.join(reports_dir, f"rl_backtest_report_{suffix}.txt")

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    print(f"\n[INFO] 📈 차트 저장: {save_fig_path}")

    model_desc = ", ".join(model_paths) if len(model_paths) > 1 else model_paths[0]
    _write_trade_report(report_path, model_desc, summary, trades)
    print(f"[INFO] 📝 리포트 저장: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Commander RL (Leverage)")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-tag", type=str, default=None,
                        help="model-source 기준 태그, e.g. lev2_seed42_001")
    parser.add_argument("--ensemble-tags", type=str, default=None,
                        help="앙상블 태그 (쉼표 구분), e.g. lev2_seed42_001,lev2_seed43_001")
    parser.add_argument("--suffix", type=str, default="",
                        help="결과 파일명 접미사")
    parser.add_argument("--leverage", type=int, default=2,
                        help="레버리지 배수 (기본 2)")
    parser.add_argument("--model-source", type=str, choices=["candidates", "runs"], default="candidates",
                        help="model-tag 해석 대상 (기본: candidates)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="모델 루트 디렉토리 (기본: root/models/commander)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="백테스트 데이터 CSV 경로 (기본: root/data/commander/base_signals_log.csv)")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="리포트/차트 출력 디렉토리 (기본: commander/reports)")
    args = parser.parse_args()

    run_rl_backtest(
        model_path=args.model_path,
        model_tag=args.model_tag,
        output_suffix=args.suffix,
        ensemble_tags=args.ensemble_tags,
        leverage=args.leverage,
        model_source=args.model_source,
        model_dir=args.model_dir,
        data_path=args.data_path,
        reports_dir=args.reports_dir,
    )
