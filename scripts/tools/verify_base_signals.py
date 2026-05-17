"""
Base Signal Verification Script
================================
학습된 base 전문가 모델의 시그널(long_score / short_score / context_score)의
예측 엣지를 RL 없이 검증한다.

수정 사항 (v2):
  1. Context Expert 로직 수정: Long/Short 모두 ctx > threshold (변동성 이벤트 확률)
  2. talib.ATR(high, low, close, 14) 사용, TP/SL 청산을 high/low wick 기준으로 검사
  3. Z-score 기준 통계량을 'train' split (없으면 'val' split)에서만 추출 (Look-ahead 제거)
  4. 4-패널 차트: Price+Markers / Long Z / Short Z / Context Score

실행 방법:
  python scripts/tools/verify_base_signals.py
  python scripts/tools/verify_base_signals.py --signal-file data/signals/BTC_USDT_signals_log.csv --split test
  python scripts/tools/verify_base_signals.py --long-z 0.8 --context-thresh 0.55 --chart-window 3000
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
import talib

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parents[2]   # crypto_bot/
SIGNALS_FILE = ROOT_DIR / "data" / "signals" / "base_signals_log.csv"
RAW_FILE     = ROOT_DIR / "data" / "raw" / "BTC_USDT_5m_raw.csv"
REPORTS_DIR  = ROOT_DIR / "reports"
OUTPUT_IMG   = REPORTS_DIR / "base_signal_verification.png"

ATR_PERIOD   = 14
TP_MULT      = 2.0      # TP = entry ± TP_MULT × ATR
SL_MULT      = 1.0      # SL = entry ∓ SL_MULT × ATR
CHART_WINDOW = 2000     # 차트에 표시할 최근 캔들 수


# ── Z-score (train 통계량 기반 transform) ────────────────────────────────────
def zscore_transform(series: pd.Series, ref_mask: pd.Series) -> pd.Series:
    """
    ref_mask에 해당하는 구간의 mean/std만 사용하여 전체 series를 정규화.
    Look-ahead bias를 방지하기 위해 train 구간 통계량만 사용.
    """
    mu    = series[ref_mask].mean()
    sigma = series[ref_mask].std(ddof=0)
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
      SHORT : short_z > short_z_thresh AND context_score > context_thresh
              (Context Expert는 방향 무관, 변동성 이벤트 발생 확률을 예측)

    청산 조건 (high/low wick 기준):
      LONG  : high[j] >= TP  → win (익절)
              low[j]  <= SL  → loss (손절)
              양쪽 동시 발생 시: open[j] <= SL 이면 gap-down loss, 아니면 win
      SHORT : low[j]  <= TP  → win
              high[j] >= SL  → loss
              양쪽 동시 발생 시: open[j] >= SL 이면 gap-up loss, 아니면 win

    반환값:
      dict with keys: trades, wins, win_rate, avg_win, avg_loss,
                      payoff_ratio, cum_return, trade_log (DataFrame)
    """
    close   = df["close"].values
    high    = df["high"].values
    low     = df["low"].values
    opn     = df["open"].values
    long_z  = df["long_z"].values
    short_z = df["short_z"].values
    ctx     = df["context_score"].values
    atr     = df["atr"].values
    n       = len(df)

    trade_log: list[dict] = []
    position  = None   # None | dict
    # position 구조: {"side", "entry", "tp", "sl", "bar"}

    for i in range(ATR_PERIOD, n):
        # ── 포지션 청산 체크 (현재 바 high/low 기준) ─────────────────────────
        if position is not None:
            h, l, o = high[i], low[i], opn[i]
            if position["side"] == "long":
                hit_tp = h >= position["tp"]
                hit_sl = l <= position["sl"]
                if hit_tp and hit_sl:
                    # 동시 발생: gap-down이면 SL 먼저
                    result = "loss" if o <= position["sl"] else "win"
                    exit_px = position["sl"] if result == "loss" else position["tp"]
                elif hit_tp:
                    result, exit_px = "win", position["tp"]
                elif hit_sl:
                    result, exit_px = "loss", position["sl"]
                else:
                    result = None

                if result is not None:
                    pnl = (exit_px - position["entry"]) / position["entry"]
                    trade_log.append({
                        "side": "long", "result": result, "pnl": pnl,
                        "entry_bar": position["bar"], "exit_bar": i,
                    })
                    position = None

            else:  # short
                hit_tp = l <= position["tp"]
                hit_sl = h >= position["sl"]
                if hit_tp and hit_sl:
                    result = "loss" if o >= position["sl"] else "win"
                    exit_px = position["sl"] if result == "loss" else position["tp"]
                elif hit_tp:
                    result, exit_px = "win", position["tp"]
                elif hit_sl:
                    result, exit_px = "loss", position["sl"]
                else:
                    result = None

                if result is not None:
                    pnl = (position["entry"] - exit_px) / position["entry"]
                    trade_log.append({
                        "side": "short", "result": result, "pnl": pnl,
                        "entry_bar": position["bar"], "exit_bar": i,
                    })
                    position = None

        # ── 신규 진입 (포지션 없을 때, 현재 바 close 기준) ──────────────────
        if position is None:
            cur_atr = atr[i]
            if cur_atr <= 0 or np.isnan(cur_atr):
                continue

            # Long: long_z > thresh AND context > thresh
            if long_z[i] > long_z_thresh and ctx[i] > context_thresh:
                entry = close[i]
                position = {
                    "side":  "long",
                    "entry": entry,
                    "tp":    entry + TP_MULT * cur_atr,
                    "sl":    entry - SL_MULT * cur_atr,
                    "bar":   i,
                }
            # Short: short_z > thresh AND context > thresh (방향 무관 변동성 필터)
            elif short_z[i] > short_z_thresh and ctx[i] > context_thresh:
                entry = close[i]
                position = {
                    "side":  "short",
                    "entry": entry,
                    "tp":    entry - TP_MULT * cur_atr,
                    "sl":    entry + SL_MULT * cur_atr,
                    "bar":   i,
                }

    # 미청산 포지션 → 마지막 바 종가로 강제 청산
    if position is not None:
        c = close[-1]
        pnl = ((c - position["entry"]) / position["entry"]
               if position["side"] == "long"
               else (position["entry"] - c) / position["entry"])
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
            "trade_log": pd.DataFrame(),
        }

    tl      = pd.DataFrame(trade_log)
    closed  = tl[tl["result"] != "open"]
    wins    = closed[closed["result"] == "win"]
    losses  = closed[closed["result"] == "loss"]

    n_wins    = len(wins)
    win_rate  = n_wins / len(closed) if len(closed) > 0 else 0.0
    avg_win   = float(wins["pnl"].mean())   if len(wins)   > 0 else 0.0
    avg_loss  = float(losses["pnl"].mean()) if len(losses) > 0 else 0.0
    payoff    = abs(avg_win / avg_loss)     if avg_loss != 0 else float("inf")
    cum_ret   = float((1 + tl["pnl"]).prod() - 1.0)

    return {
        "trades":       len(tl),
        "wins":         n_wins,
        "win_rate":     win_rate,
        "avg_win":      avg_win,
        "avg_loss":     avg_loss,
        "payoff_ratio": payoff,
        "cum_return":   cum_ret,
        "trade_log":    tl,
    }


# ── 4-패널 차트 생성 ──────────────────────────────────────────────────────────
def plot_signals(
    df: pd.DataFrame,
    trade_log: pd.DataFrame,
    output_path: Path,
    context_thresh: float,
    window: int = CHART_WINDOW,
) -> None:
    """
    4-패널 차트:
      Panel 1: Close 가격 + Long(^) / Short(v) 진입 마커
      Panel 2: Long Z-score  (진입 기준선 포함)
      Panel 3: Short Z-score (진입 기준선 포함)
      Panel 4: Context Score (threshold 기준선 포함)
    """
    vis       = df.iloc[-window:].copy()
    vis_lo    = len(df) - window          # df 전체 기준 vis 시작 인덱스

    close_vals = vis["close"].values
    long_z_v   = vis["long_z"].values
    short_z_v  = vis["short_z"].values
    ctx_v      = vis["context_score"].values
    x          = np.arange(len(vis))

    # vis 구간 내 진입 trades 필터링 (df 전역 인덱스 기준)
    if len(trade_log) > 0:
        vis_trades = trade_log[trade_log["entry_bar"] >= vis_lo].copy()
        vis_trades = vis_trades[vis_trades["entry_bar"] < len(df)]
    else:
        vis_trades = pd.DataFrame()

    # ── Figure 설정 ──────────────────────────────────────────────────────────
    DARK_BG   = "#0d1117"
    PANEL_BG  = "#161b22"
    BORDER_C  = "#30363d"
    TICK_C    = "white"

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[3, 1, 1, 1],
        hspace=0.06,
    )

    def style_ax(ax: plt.Axes, show_xticks: bool = False) -> None:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TICK_C, labelbottom=show_xticks)
        for sp in ax.spines.values():
            sp.set_color(BORDER_C)
        ax.yaxis.label.set_color(TICK_C)
        ax.xaxis.label.set_color(TICK_C)

    # ── Panel 1: 가격 + 진입 마커 ────────────────────────────────────────────
    ax_price = fig.add_subplot(gs[0])
    style_ax(ax_price)
    ax_price.plot(x, close_vals, color="#58a6ff", linewidth=0.8, label="Close")
    ax_price.set_ylabel("Close (USDT)")
    ax_price.set_title(
        f"Base Signal Verification  (last {window} bars) | "
        f"TP={TP_MULT}×ATR14  SL={SL_MULT}×ATR14",
        color="white", fontsize=11, pad=8,
    )

    _long_plotted  = False
    _short_plotted = False
    if len(vis_trades) > 0:
        for _, row in vis_trades.iterrows():
            xi = int(row["entry_bar"]) - vis_lo
            if not (0 <= xi < len(vis)):
                continue
            yi = close_vals[xi]
            if row["side"] == "long":
                lbl = "Long entry" if not _long_plotted else ""
                ax_price.scatter(xi, yi, marker="^", color="#3fb950",
                                 s=55, zorder=5, label=lbl)
                _long_plotted = True
            else:
                lbl = "Short entry" if not _short_plotted else ""
                ax_price.scatter(xi, yi, marker="v", color="#f85149",
                                 s=55, zorder=5, label=lbl)
                _short_plotted = True

    ax_price.legend(facecolor=PANEL_BG, labelcolor="white",
                    fontsize=8, loc="upper left")

    # ── Panel 2: Long Z-score ────────────────────────────────────────────────
    ax_lz = fig.add_subplot(gs[1], sharex=ax_price)
    style_ax(ax_lz)
    ax_lz.plot(x, long_z_v, color="#3fb950", linewidth=0.7, label="Long Z")
    ax_lz.axhline(1.0, color="#3fb950", linewidth=0.6, linestyle="--", alpha=0.6,
                  label="Entry threshold")
    ax_lz.axhline(0.0, color="#8b949e", linewidth=0.4, linestyle="-", alpha=0.35)
    ax_lz.set_ylabel("Long Z")
    ax_lz.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=7, loc="upper left")

    # ── Panel 3: Short Z-score ───────────────────────────────────────────────
    ax_sz = fig.add_subplot(gs[2], sharex=ax_price)
    style_ax(ax_sz)
    ax_sz.plot(x, short_z_v, color="#f85149", linewidth=0.7, label="Short Z")
    ax_sz.axhline(1.0, color="#f85149", linewidth=0.6, linestyle="--", alpha=0.6,
                  label="Entry threshold")
    ax_sz.axhline(0.0, color="#8b949e", linewidth=0.4, linestyle="-", alpha=0.35)
    ax_sz.set_ylabel("Short Z")
    ax_sz.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=7, loc="upper left")

    # ── Panel 4: Context Score ───────────────────────────────────────────────
    ax_ctx = fig.add_subplot(gs[3], sharex=ax_price)
    style_ax(ax_ctx, show_xticks=True)
    ax_ctx.plot(x, ctx_v, color="#e3b341", linewidth=0.7, label="Context Score")
    ax_ctx.axhline(
        context_thresh, color="#e3b341", linewidth=0.8, linestyle="--", alpha=0.8,
        label=f"Threshold ({context_thresh:.2f})",
    )
    ax_ctx.set_ylabel("Context")
    ax_ctx.set_xlabel("Bar index (recent)", color="white", fontsize=8)
    ax_ctx.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=7, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[차트 저장] {output_path}")


# ── 결과 출력 ─────────────────────────────────────────────────────────────────
def print_results(
    result: dict,
    split: str,
    long_z: float,
    short_z: float,
    ctx: float,
    ref_split: str,
) -> None:
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  Base Signal Rule-based Backtest ({split.upper()} split)")
    print(f"  Z-score 기준 통계량 : {ref_split.upper()} split")
    print(f"  진입 조건 (LONG)    : long_z > {long_z:.2f}  AND  ctx > {ctx:.2f}")
    print(f"  진입 조건 (SHORT)   : short_z > {short_z:.2f} AND  ctx > {ctx:.2f}")
    print(f"  청산 조건           : TP={TP_MULT}×ATR14 (high/low)  /  SL={SL_MULT}×ATR14")
    print(sep)

    if result["trades"] == 0:
        print("  ⚠  조건을 만족하는 진입이 없습니다.")
        print(f"{sep}\n")
        return

    expectancy = (result["win_rate"] * result["avg_win"]
                  + (1 - result["win_rate"]) * result["avg_loss"])
    edge_mark  = "✅ 양의 기대값" if expectancy > 0 else "❌ 음의 기대값"

    long_t  = result["trade_log"][result["trade_log"]["side"] == "long"]
    short_t = result["trade_log"][result["trade_log"]["side"] == "short"]

    print(f"  {'항목':<20} {'전체':>8}  {'LONG':>8}  {'SHORT':>8}")
    print(f"  {'─'*20} {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'총 진입 횟수':<20} {result['trades']:>8d}"
          f"  {len(long_t):>8d}  {len(short_t):>8d}")

    def wr(sub):
        cl = sub[sub["result"] != "open"]
        if len(cl) == 0: return 0.0
        return len(cl[cl["result"] == "win"]) / len(cl) * 100

    print(f"  {'승률':<20} {result['win_rate']*100:>7.1f}%"
          f"  {wr(long_t):>7.1f}%  {wr(short_t):>7.1f}%")
    print(f"  {'평균 이익':<20} {result['avg_win']*100:>+7.3f}%")
    print(f"  {'평균 손실':<20} {result['avg_loss']*100:>+7.3f}%")
    print(f"  {'손익비 (Payoff)':<20} {result['payoff_ratio']:>8.2f}")
    print(f"  {'누적 수익률':<20} {result['cum_return']*100:>+7.2f}%")
    print(f"  {'기대값 (1건)':<20} {expectancy*100:>+7.3f}%")
    print(f"  {'엣지 판정':<20}  {edge_mark}")
    print(sep)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Base 전문가 시그널 규칙 기반 검증 스크립트 (v2)"
    )
    parser.add_argument(
        "--signal-file",
        type=Path,
        default=SIGNALS_FILE,
        help=f"시그널 CSV 경로 (기본값: {SIGNALS_FILE.relative_to(ROOT_DIR)})",
    )
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=RAW_FILE,
        help=f"Raw OHLCV CSV 경로 (기본값: {RAW_FILE.relative_to(ROOT_DIR)})",
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

    # ── 시그널 로드 ───────────────────────────────────────────────────────────
    if not args.signal_file.exists():
        print(f"[ERROR] 시그널 파일이 없습니다: {args.signal_file}")
        print("        먼저 scripts/02_extract_signals.py 를 실행하세요.")
        sys.exit(1)

    print(f"[로드] 시그널: {args.signal_file}")
    sig = pd.read_csv(args.signal_file, parse_dates=["datetime"])
    sig.sort_values("datetime", inplace=True)
    sig.reset_index(drop=True, inplace=True)

    # ── Raw OHLCV 로드 및 병합 ────────────────────────────────────────────────
    if not args.raw_file.exists():
        print(f"[ERROR] Raw OHLCV 파일이 없습니다: {args.raw_file}")
        sys.exit(1)

    print(f"[로드] Raw OHLCV: {args.raw_file}")
    raw = pd.read_csv(args.raw_file,
                      usecols=["datetime", "open", "high", "low", "close"])
    # timezone 제거 후 naive datetime으로 통일
    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True).dt.tz_localize(None)
    raw.sort_values("datetime", inplace=True)

    df = pd.merge(
        sig,
        raw.rename(columns={
            "open":  "open",
            "high":  "high",
            "low":   "low",
            "close": "close_raw",  # signals의 close와 구분
        }),
        on="datetime",
        how="left",
    )

    # signals의 close는 모델 추론 시 사용된 값, raw의 close와 동일 → raw 값 우선
    df["close"] = df["close_raw"].fillna(df["close"])
    df.drop(columns=["close_raw"], inplace=True)

    missing = df["high"].isna().sum()
    if missing > 0:
        print(f"[WARNING] raw 병합 후 high/low 누락 {missing}행 → ffill 처리")
        df[["open", "high", "low"]] = df[["open", "high", "low"]].ffill()

    # ── talib ATR 계산 ────────────────────────────────────────────────────────
    df["atr"] = talib.ATR(
        df["high"].values,
        df["low"].values,
        df["close"].values,
        timeperiod=ATR_PERIOD,
    )

    # ── Z-score (train 통계량 기반, Look-ahead 없음) ──────────────────────────
    splits_available = set(df["split"].dropna().unique())
    if "train" in splits_available:
        ref_split = "train"
    elif "val" in splits_available:
        # base_signals_log.csv 처럼 train split이 없을 때 val을 기준으로 사용
        ref_split = "val"
        print("[INFO] 'train' split이 없어 'val' split 통계량을 Z-score 기준으로 사용합니다.")
    else:
        ref_split = "all"
        print("[WARNING] train/val split 없음. 전체 데이터 통계량 사용 (look-ahead 가능)")

    ref_mask      = (df["split"] == ref_split) if ref_split != "all" else pd.Series([True] * len(df), index=df.index)
    df["long_z"]  = zscore_transform(df["long_score"],  ref_mask)
    df["short_z"] = zscore_transform(df["short_score"], ref_mask)
    print(f"[Z-score] 기준 split={ref_split}  "
          f"long_score: μ={df.loc[ref_mask,'long_score'].mean():.5f}  "
          f"σ={df.loc[ref_mask,'long_score'].std():.5f}")

    # ── split 필터링 (백테스트 대상) ─────────────────────────────────────────
    if args.split == "all":
        df_bt    = df.copy()
        bt_offset = 0
    else:
        mask      = df["split"] == args.split
        bt_offset = int(mask.values.argmax())   # df 전역 인덱스 시작점
        df_bt     = df[mask].copy()
        df_bt.reset_index(drop=True, inplace=True)

    if len(df_bt) == 0:
        print(f"[ERROR] split='{args.split}' 데이터가 없습니다.")
        sys.exit(1)

    print(f"[백테스트] split={args.split}  행={len(df_bt):,}  "
          f"기간: {df_bt['datetime'].iloc[0]} ~ {df_bt['datetime'].iloc[-1]}")

    # ── 1. Rule-based 백테스트 ────────────────────────────────────────────────
    result = run_backtest(
        df             = df_bt,
        long_z_thresh  = args.long_z,
        short_z_thresh = args.short_z,
        context_thresh = args.context_thresh,
    )
    print_results(
        result, args.split, args.long_z, args.short_z,
        args.context_thresh, ref_split,
    )

    # entry_bar (df_bt 로컬 인덱스) → df 전역 인덱스로 변환
    tl = result["trade_log"]
    if isinstance(tl, pd.DataFrame) and len(tl) > 0:
        tl = tl.copy()
        tl["entry_bar"] = tl["entry_bar"] + bt_offset

    # ── 2. 4-패널 차트 생성 ──────────────────────────────────────────────────
    plot_signals(
        df             = df,
        trade_log      = tl if isinstance(tl, pd.DataFrame) else pd.DataFrame(tl),
        output_path    = OUTPUT_IMG,
        context_thresh = args.context_thresh,
        window         = args.chart_window,
    )


if __name__ == "__main__":
    main()
