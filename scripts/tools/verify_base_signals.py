"""
Base Signal Verification Script
================================
학습된 base 전문가 모델이 뱉은 시그널(long_score / short_score / context_score)의
예측 엣지를 RL 없이 검증한다.

실행 방법:
  python scripts/tools/verify_base_signals.py
  python scripts/tools/verify_base_signals.py --split test --long-z 1.0 --short-z 1.0 --context-thresh 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 대응
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]   # crypto_bot/
SIGNALS_FILE = ROOT_DIR / "data" / "signals" / "base_signals_log.csv"
REPORTS_DIR  = ROOT_DIR / "reports"
OUTPUT_IMG   = REPORTS_DIR / "base_signal_verification.png"

ATR_PERIOD   = 14
TP_MULT      = 2.0      # TP = entry ± TP_MULT * ATR
SL_MULT      = 1.0      # SL = entry ∓ SL_MULT * ATR
CHART_WINDOW = 2000     # 차트에 표시할 최근 캔들 수


# ── ATR (close-only 근사) ──────────────────────────────────────────────────────
def compute_atr(close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """High/Low 없이 |close_t - close_{t-1}| 의 rolling mean 으로 ATR 근사."""
    tr = close.diff().abs()
    return tr.rolling(period, min_periods=1).mean()


# ── Z-score 정규화 ────────────────────────────────────────────────────────────
def zscore(series: pd.Series) -> pd.Series:
    mu, sigma = series.mean(), series.std(ddof=0)
    if sigma == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sigma


# ── Rule-based 백테스트 ───────────────────────────────────────────────────────
def run_backtest(
    df: pd.DataFrame,
    long_z_thresh: float,
    short_z_thresh: float,
    context_thresh: float,
) -> dict:
    """
    진입 조건:
      LONG  : long_z  > long_z_thresh  AND context_score > context_thresh
      SHORT : short_z > short_z_thresh AND context_score < context_thresh

    청산 조건 (5분봉 close 기준):
      LONG  : close >= entry + TP_MULT*ATR  (win)
             close <= entry - SL_MULT*ATR  (loss)
      SHORT : close <= entry - TP_MULT*ATR  (win)
             close >= entry + SL_MULT*ATR  (loss)

    반환값:
      dict with keys: trades, wins, win_rate, avg_win, avg_loss,
                      payoff_ratio, cum_return, trade_log
    """
    close   = df["close"].values
    long_z  = df["long_z"].values
    short_z = df["short_z"].values
    ctx     = df["context_score"].values
    atr     = df["atr"].values

    trade_log: list[dict] = []
    position  = None   # None | {"side": "long"/"short", "entry": float, "tp": float, "sl": float, "bar": int}
    n = len(df)

    for i in range(ATR_PERIOD, n):
        # ── 포지션 관리 (먼저 청산 체크) ──
        if position is not None:
            c = close[i]
            if position["side"] == "long":
                if c >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) / position["entry"]
                    trade_log.append({"side": "long", "result": "win", "pnl": pnl,
                                      "entry_bar": position["bar"], "exit_bar": i})
                    position = None
                elif c <= position["sl"]:
                    pnl = (position["sl"] - position["entry"]) / position["entry"]
                    trade_log.append({"side": "long", "result": "loss", "pnl": pnl,
                                      "entry_bar": position["bar"], "exit_bar": i})
                    position = None
            else:  # short
                if c <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) / position["entry"]
                    trade_log.append({"side": "short", "result": "win", "pnl": pnl,
                                      "entry_bar": position["bar"], "exit_bar": i})
                    position = None
                elif c >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) / position["entry"]
                    trade_log.append({"side": "short", "result": "loss", "pnl": pnl,
                                      "entry_bar": position["bar"], "exit_bar": i})
                    position = None

        # ── 신규 진입 (포지션 없을 때만) ──
        if position is None:
            cur_atr = atr[i]
            if cur_atr <= 0:
                continue

            if long_z[i] > long_z_thresh and ctx[i] > context_thresh:
                entry = close[i]
                position = {
                    "side":  "long",
                    "entry": entry,
                    "tp":    entry + TP_MULT * cur_atr,
                    "sl":    entry - SL_MULT * cur_atr,
                    "bar":   i,
                }
            elif short_z[i] > short_z_thresh and ctx[i] < context_thresh:
                entry = close[i]
                position = {
                    "side":  "short",
                    "entry": entry,
                    "tp":    entry - TP_MULT * cur_atr,
                    "sl":    entry + SL_MULT * cur_atr,
                    "bar":   i,
                }

    # 미청산 포지션 → 마지막 bar 종가로 강제 청산
    if position is not None:
        c = close[-1]
        if position["side"] == "long":
            pnl = (c - position["entry"]) / position["entry"]
        else:
            pnl = (position["entry"] - c) / position["entry"]
        trade_log.append({
            "side":      position["side"],
            "result":    "open",
            "pnl":       pnl,
            "entry_bar": position["bar"],
            "exit_bar":  n - 1,
        })

    if not trade_log:
        return {
            "trades": 0, "wins": 0, "win_rate": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "payoff_ratio": 0.0, "cum_return": 0.0,
            "trade_log": [],
        }

    tl      = pd.DataFrame(trade_log)
    closed  = tl[tl["result"] != "open"]
    wins    = closed[closed["result"] == "win"]
    losses  = closed[closed["result"] == "loss"]

    total      = len(tl)
    n_wins     = len(wins)
    win_rate   = n_wins / len(closed) if len(closed) > 0 else 0.0
    avg_win    = wins["pnl"].mean()    if len(wins)   > 0 else 0.0
    avg_loss   = losses["pnl"].mean()  if len(losses) > 0 else 0.0
    payoff     = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    cum_ret    = (1 + tl["pnl"]).prod() - 1.0

    return {
        "trades":       total,
        "wins":         n_wins,
        "win_rate":     win_rate,
        "avg_win":      avg_win,
        "avg_loss":     avg_loss,
        "payoff_ratio": payoff,
        "cum_return":   cum_ret,
        "trade_log":    tl,
    }


# ── 차트 생성 ─────────────────────────────────────────────────────────────────
def plot_signals(
    df: pd.DataFrame,
    trade_log: pd.DataFrame,
    output_path: Path,
    window: int = CHART_WINDOW,
) -> None:
    """최근 window 캔들 구간을 시각화하고 진입 타점을 마커로 표시."""
    vis = df.iloc[-window:].copy()
    vis_start = vis.index[0]

    # 차트 구간 내 진입 trade만 추출
    if len(trade_log) > 0:
        vis_trades = trade_log[trade_log["entry_bar"] >= (len(df) - window)]
    else:
        vis_trades = pd.DataFrame()

    close_vals  = vis["close"].values
    long_z_vals = vis["long_z"].values
    short_z_vals= vis["short_z"].values
    x           = np.arange(len(vis))

    # 인덱스 → x 위치 매핑 (전체 df 기준 entry_bar → vis 상대 위치)
    def bar_to_x(global_bar: int) -> int:
        return global_bar - (len(df) - window)

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.08)

    # ── 상단: Close + 진입 마커 ───────────────────────────────────────────────
    ax_price = fig.add_subplot(gs[0])
    ax_price.set_facecolor("#161b22")
    ax_price.plot(x, close_vals, color="#58a6ff", linewidth=0.8, label="Close")
    ax_price.set_ylabel("Close Price (USDT)", color="white")
    ax_price.tick_params(colors="white", labelbottom=False)
    ax_price.spines["bottom"].set_color("#30363d")
    ax_price.spines["top"].set_color("#30363d")
    ax_price.spines["left"].set_color("#30363d")
    ax_price.spines["right"].set_color("#30363d")

    # 진입 마커
    if len(vis_trades) > 0:
        for _, row in vis_trades.iterrows():
            xi = bar_to_x(int(row["entry_bar"]))
            if 0 <= xi < len(vis):
                yi = close_vals[xi]
                if row["side"] == "long":
                    ax_price.scatter(xi, yi, marker="^", color="#3fb950",
                                     s=60, zorder=5, label="Long entry" if "Long entry" not in [l.get_label() for l in ax_price.lines + ax_price.collections] else "")
                else:
                    ax_price.scatter(xi, yi, marker="v", color="#f85149",
                                     s=60, zorder=5, label="Short entry" if "Short entry" not in [l.get_label() for l in ax_price.lines + ax_price.collections] else "")

    # 중복 레전드 제거
    handles, labels = ax_price.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax_price.legend(seen.values(), seen.keys(), facecolor="#161b22",
                    labelcolor="white", fontsize=8, loc="upper left")
    ax_price.set_title(
        f"Base Signal Verification  (last {window} bars) | "
        f"TP={TP_MULT}×ATR  SL={SL_MULT}×ATR",
        color="white", fontsize=11, pad=8,
    )

    # ── 중단: Long Z-score ───────────────────────────────────────────────────
    ax_long = fig.add_subplot(gs[1], sharex=ax_price)
    ax_long.set_facecolor("#161b22")
    ax_long.plot(x, long_z_vals, color="#3fb950", linewidth=0.7, label="Long Z")
    ax_long.axhline(1.0, color="#3fb950", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_long.axhline(0.0, color="#8b949e", linewidth=0.4, linestyle="-", alpha=0.4)
    ax_long.set_ylabel("Long Z", color="white", fontsize=8)
    ax_long.tick_params(colors="white", labelbottom=False)
    for sp in ax_long.spines.values():
        sp.set_color("#30363d")
    ax_long.legend(facecolor="#161b22", labelcolor="white", fontsize=7, loc="upper left")

    # ── 하단: Short Z-score ──────────────────────────────────────────────────
    ax_short = fig.add_subplot(gs[2], sharex=ax_price)
    ax_short.set_facecolor("#161b22")
    ax_short.plot(x, short_z_vals, color="#f85149", linewidth=0.7, label="Short Z")
    ax_short.axhline(1.0, color="#f85149", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_short.axhline(0.0, color="#8b949e", linewidth=0.4, linestyle="-", alpha=0.4)
    ax_short.set_ylabel("Short Z", color="white", fontsize=8)
    ax_short.set_xlabel("Bar index (recent)", color="white", fontsize=8)
    ax_short.tick_params(colors="white")
    for sp in ax_short.spines.values():
        sp.set_color("#30363d")
    ax_short.legend(facecolor="#161b22", labelcolor="white", fontsize=7, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[차트 저장] {output_path}")


# ── 결과 출력 ─────────────────────────────────────────────────────────────────
def print_results(result: dict, split: str, long_z: float, short_z: float, ctx: float) -> None:
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Base Signal Rule-based Backtest ({split.upper()} split)")
    print(f"  진입 조건: Long_Z > {long_z:.1f} & ctx > {ctx:.2f}")
    print(f"            Short_Z > {short_z:.1f} & ctx < {ctx:.2f}")
    print(f"  청산 조건: TP={TP_MULT}×ATR14  /  SL={SL_MULT}×ATR14")
    print(sep)
    if result["trades"] == 0:
        print("  ⚠  조건을 만족하는 진입이 없습니다.")
        print(f"{sep}\n")
        return

    print(f"  총 진입 횟수  : {result['trades']:>6d}")
    print(f"  승  (TP 도달) : {result['wins']:>6d}  ({result['win_rate']*100:.1f} %)")
    print(f"  평균 이익     : {result['avg_win']*100:>+.3f} %")
    print(f"  평균 손실     : {result['avg_loss']*100:>+.3f} %")
    print(f"  손익비 (Payoff): {result['payoff_ratio']:>6.2f}")
    print(f"  누적 수익률   : {result['cum_return']*100:>+.2f} %")

    # Edge 간이 판정
    expectancy = result["win_rate"] * result["avg_win"] + (1 - result["win_rate"]) * result["avg_loss"]
    print(f"  기대값 (1건)  : {expectancy*100:>+.3f} %")
    edge = "✅ 양의 기대값" if expectancy > 0 else "❌ 음의 기대값"
    print(f"  엣지 판정     : {edge}")
    print(sep)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Base 전문가 시그널 규칙 기반 검증 스크립트"
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "all"],
        default="test",
        help="검증에 사용할 split (기본값: test)",
    )
    parser.add_argument(
        "--long-z",
        type=float,
        default=1.0,
        help="Long 진입 Z-score 임계값 (기본값: 1.0)",
    )
    parser.add_argument(
        "--short-z",
        type=float,
        default=1.0,
        help="Short 진입 Z-score 임계값 (기본값: 1.0)",
    )
    parser.add_argument(
        "--context-thresh",
        type=float,
        default=0.5,
        help="Context score 임계값 (기본값: 0.5)",
    )
    parser.add_argument(
        "--chart-window",
        type=int,
        default=CHART_WINDOW,
        help=f"차트 표시 캔들 수 (기본값: {CHART_WINDOW})",
    )
    args = parser.parse_args()

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    if not SIGNALS_FILE.exists():
        print(f"[ERROR] 시그널 파일이 없습니다: {SIGNALS_FILE}")
        print("        먼저 scripts/02_extract_signals.py 를 실행하세요.")
        sys.exit(1)

    print(f"[로드] {SIGNALS_FILE}")
    df_full = pd.read_csv(SIGNALS_FILE, parse_dates=["datetime"])
    df_full.sort_values("datetime", inplace=True)
    df_full.reset_index(drop=True, inplace=True)

    # Z-score는 전체 데이터로 계산 (look-ahead 없이 표준화 기준만 전체 사용)
    df_full["long_z"]  = zscore(df_full["long_score"])
    df_full["short_z"] = zscore(df_full["short_score"])
    df_full["atr"]     = compute_atr(df_full["close"], ATR_PERIOD)

    # ── split 필터링 ──────────────────────────────────────────────────────────
    if args.split == "all":
        df_bt = df_full.copy()
        bt_offset = 0
    else:
        mask = df_full["split"] == args.split
        bt_offset = int(mask.values.argmax())   # df_full 내 시작 인덱스
        df_bt = df_full[mask].copy()
        df_bt.reset_index(drop=True, inplace=True)

    print(f"[split={args.split}]  행 수: {len(df_bt):,}  "
          f"기간: {df_bt['datetime'].iloc[0]} ~ {df_bt['datetime'].iloc[-1]}")

    # ── 1. Rule-based 백테스트 ────────────────────────────────────────────────
    result = run_backtest(
        df      = df_bt,
        long_z_thresh  = args.long_z,
        short_z_thresh = args.short_z,
        context_thresh = args.context_thresh,
    )
    print_results(result, args.split, args.long_z, args.short_z, args.context_thresh)

    # entry_bar (df_bt 기준) → df_full 기준으로 변환
    tl = result["trade_log"]
    if isinstance(tl, pd.DataFrame) and len(tl) > 0:
        tl = tl.copy()
        tl["entry_bar"] = tl["entry_bar"] + bt_offset

    # ── 2. 차트 생성 (전체 df 기준 마지막 window 캔들) ────────────────────────
    plot_signals(
        df         = df_full,
        trade_log  = tl if isinstance(tl, pd.DataFrame) else pd.DataFrame(tl),
        output_path= OUTPUT_IMG,
        window     = args.chart_window,
    )


if __name__ == "__main__":
    main()
