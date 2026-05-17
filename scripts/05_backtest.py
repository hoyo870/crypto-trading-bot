"""
단일 모델 백테스트 실행 엔진.

역할:
  - 모델 1개(또는 앙상블 경로 목록)를 로드해 전체 구간을 시뮬레이션
  - 수익률, MDD, Sharpe, 승률, 청산 여부를 계산하고 JSON + 차트로 저장
  - 랭킹/배치 관리는 담당하지 않음 → 06_rank.py / 07_backtest_batch.py 참조

단독 실행 예시:
  python scripts/05_backtest.py \\
      --model-path checkpoints/rl_generations/gen1/lev1_stb_seed12345_001/final_model_lev1_stb_seed12345_001.zip \\
      --tag lev1_stb_seed12345_001 \\
      --leverage 1 \\
      --reports-dir reports/gen1
"""

import os
import sys
import argparse
import json
import re
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import re
import numpy as np

from sb3_contrib import MaskablePPO

# ── 경로 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.envs.trading_env_baby import BabyLeverageTradingEnv as LeverageTradingEnv

# ── 로깅 ───────────────────────────────────────────────────────────────────
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

_custom_log_dir = os.environ.get("CUSTOM_LOG_DIR")
if _custom_log_dir:
    os.makedirs(_custom_log_dir, exist_ok=True)
    logging.getLogger().addHandler(
        logging.FileHandler(os.path.join(_custom_log_dir, "backtest.log"), encoding='utf-8')
    )


# ── 유틸리티 ───────────────────────────────────────────────────────────────
def _infer_leverage_from_tag(tag: str):
    """태그 내 어디서든 lev(\\d+)_ 패턴을 탐색합니다 (parent_ 접두사 포함 대응)."""
    m = re.search(r"lev(\d+)_", str(tag).strip().lower())
    return int(m.group(1)) if m else None


def _infer_profile_from_tag(tag: str) -> str:
    """태그 내 _stb_ / _bal_ / _agg_ 를 탐색해 tuning_profile 추론."""
    t = str(tag).lower()
    if "_stb" in t: return "stable"
    if "_agg" in t: return "aggressive"
    return "balanced"  # 기본값


def _calc_mdd(balances):
    arr = np.array(balances, dtype=float)
    peak = np.maximum.accumulate(arr)
    return float(((arr - peak) / peak).min() * 100.0)


def _calc_sharpe(balances, steps_per_year=105120):
    arr = np.array(balances, dtype=float)
    if len(arr) < 2:
        return 0.0
    rets = np.diff(arr) / arr[:-1]
    sigma = rets.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float(rets.mean() / sigma * np.sqrt(steps_per_year))


def _load_datetimes(data_path: str):
    df = pd.read_csv(data_path, usecols=["datetime"])
    return pd.to_datetime(df["datetime"], errors="coerce").reset_index(drop=True)


def _resolve_model_path(tag: str, model_dir: str):
    """태그 → 모델 .zip 경로 자동 탐색."""
    clean = tag[:-4] if tag.endswith(".zip") else tag
    folder = os.path.join(model_dir, clean)
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.endswith(".zip") and ("final" in f or "best" in f):
                return os.path.join(folder, f)
    direct = os.path.join(model_dir, f"{clean}.zip")
    if os.path.exists(direct):
        return direct
    return None


# ── 핵심 백테스트 함수 ─────────────────────────────────────────────────────
def run_backtest(model_paths: list, tag: str, leverage: int,
                 data_path: str, reports_dir: str,
                 tuning_profile: str = "balanced") -> dict:
    """
    모델 1개(또는 앙상블)를 환경 전체 구간으로 백테스트합니다.

    Returns:
        summary dict (JSON + 차트 저장 후 반환)
    """
    dt_index = _load_datetimes(data_path)
    env = LeverageTradingEnv(data_path=data_path, leverage=leverage,
                             tuning_profile=tuning_profile)

    models = [MaskablePPO.load(p) for p in model_paths]
    obs, _ = env.reset(options={"start_step": 0, "max_ep_steps": None})
    # ── 전구간(Full Interval) 무결성 보장 ──────────────────────────────────
    # max_ep_steps=None 으로 에피소드 길이 제한 없이 데이터 처음부터 끝까지 실행
    assert env.max_episode_steps is None, "max_episode_steps 가 None 이어야 합니다"

    balances = [env.balance]
    liq_steps = []
    done = False

    while not done:
        pre_step     = env.current_step
        pre_position = env.position

        # 앙상블: 단순 다수결
        votes = []
        for m in models:
            # action_masks 메서드가 있으면 마스킹 적용
            masks = env.action_masks() if hasattr(env, "action_masks") else None
            action, _ = m.predict(obs, deterministic=True, action_masks=masks)
            votes.append(int(action) if np.ndim(action) == 0 else int(action[0]))

        if len(votes) == 1:
            act = votes[0]
        else:
            counts = {a: votes.count(a) for a in set(votes)}
            act = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

        obs, _, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        balances.append(env.balance)

        if info.get("liquidated") and pre_position != 0:
            liq_steps.append(pre_step)

    final_balance = info.get("final_balance", env.balance)
    summary = {
        "tag":              tag,
        "leverage":         leverage,
        "initial_balance":  env.initial_balance,
        "final_balance":    final_balance,
        "total_return_pct": (final_balance - env.initial_balance) / env.initial_balance * 100,
        "mdd_pct":          _calc_mdd(balances),
        "sharpe_ratio":     _calc_sharpe(balances),
        "total_trades":     info.get("total_trades", 0),
        "win_rate":         info.get("win_rate", 0.0),
        "liquidated":       info.get("liquidated", False),
    }

    # ── 차트 저장 ─────────────────────────────────────────────────────────
    os.makedirs(reports_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(dt_index.iloc[:len(balances)], balances,
             label=f"{tag} ({leverage}x)", color="blue")
    plt.axhline(env.initial_balance, color="red", linestyle="--")
    if liq_steps:
        liq_times = [dt_index.iloc[s] for s in liq_steps if s < len(dt_index)]
        liq_bals  = [balances[min(s, len(balances) - 1)] for s in liq_steps]
        plt.scatter(liq_times, liq_bals, color="red", s=60, zorder=5, label="Liquidation")
    plt.title(f"Backtest: {tag}", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(reports_dir, f"chart_{tag}.png"), dpi=200)
    plt.close()

    # ── JSON 저장 ─────────────────────────────────────────────────────────
    with open(os.path.join(reports_dir, f"summary_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단일 모델 백테스트")

    parser.add_argument("--model-path", type=str, required=True,
                        help="모델 .zip 경로 (쉼표로 구분 시 앙상블)")
    parser.add_argument("--tag", type=str, default=None,
                        help="모델 태그 (미지정 시 파일명 자동 유도)")
    parser.add_argument("--leverage", type=int, default=None,
                        help="레버리지 (미지정 시 태그에서 자동 유도)")
    parser.add_argument("--data-path", type=str,
                        default=os.path.join(ROOT_DIR, "data", "signals", "base_signals_log.csv"))
    parser.add_argument("--reports-dir", type=str,
                        default=os.path.join(ROOT_DIR, "reports"))
    parser.add_argument("--tuning-profile",
                        choices=["stable", "balanced", "aggressive"], default="balanced")

    args = parser.parse_args()

    model_paths = [p.strip() for p in args.model_path.split(",") if p.strip()]
    tag      = args.tag or os.path.splitext(os.path.basename(model_paths[0]))[0]
    leverage = args.leverage or _infer_leverage_from_tag(tag) or 2

    logger.info(f"백테스트 시작: {tag} (lev={leverage}x)")
    result = run_backtest(model_paths, tag, leverage,
                          args.data_path, args.reports_dir, args.tuning_profile)
    logger.info(
        f"완료: 수익률={result['total_return_pct']:+.2f}% | "
        f"MDD={result['mdd_pct']:.2f}% | Sharpe={result['sharpe_ratio']:.3f}"
    )
