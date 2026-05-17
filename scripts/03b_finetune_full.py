"""
03b_finetune_full.py — Baby 환경 우승 가중치 → Full 환경 커리큘럼 파인튜닝

사용 예시:
  python scripts/03b_finetune_full.py \
      --baby-model-path models/gen1/best_gen1_lev2_bal_seed42_001.zip \
      --leverage 2 \
      --tuning-profile balanced \
      --data-path data/BTC_USDT_processed.csv \
      --timesteps 500000 \
      --model-dir models/finetuned \
      --log-dir logs/finetune
"""

import os
import sys
import time
import argparse
import logging

import numpy as np
import torch
import pandas as pd

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ── 경로 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 03_train_rl.py 에서 콜백/프로파일 재사용
from scripts.train_rl_components import (  # noqa: E402 — fallback import
    CustomEvalCallback, SmartStopCallback, PPO_TUNING_PROFILES,
)

from src.envs.trading_env import LeverageTradingEnv  # noqa: E402

# ── 로깅 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("finetune_full")

# ── 파인튜닝 고정 하이퍼파라미터 ─────────────────────────────────────────
FINETUNE_LR       = 1e-5   # 보수적 학습률 (catastrophic forgetting 방지)
FINETUNE_ENT_COEF = 0.003  # 보수적 엔트로피 계수


def _mask_fn(env):
    return env.action_masks()


def _make_env(df: pd.DataFrame, leverage: int, mode: str):
    """ActionMasker 래핑된 LeverageTradingEnv 생성."""
    e = LeverageTradingEnv(df=df, leverage=leverage, mode=mode)
    return Monitor(ActionMasker(e, _mask_fn))


def finetune(
    baby_model_path: str,
    leverage: int,
    tuning_profile: str,
    data_path: str,
    total_timesteps: int,
    model_dir: str,
    log_dir: str,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 5,
    patience: int = 10,
    reward_target: float = 5.0,
):
    """Baby 가중치를 Full 환경에 이식하여 파인튜닝."""
    assert os.path.isfile(baby_model_path), f"Baby 모델을 찾을 수 없습니다: {baby_model_path}"

    # ── 태그 생성 ──────────────────────────────────────────────────────────
    from pathlib import Path
    baby_stem = Path(baby_model_path).stem
    tag = f"finetune_lev{leverage}_{tuning_profile}_{baby_stem}"
    save_dir = os.path.join(model_dir, tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    logger.info(f"=== Curriculum Transfer Finetune ===")
    logger.info(f"  Baby model : {baby_model_path}")
    logger.info(f"  Tag        : {tag}")
    logger.info(f"  Leverage   : {leverage}x | Profile: {tuning_profile}")
    logger.info(f"  Timesteps  : {total_timesteps:,}")

    # ── 데이터 로드 ────────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    logger.info(f"  CSV 로드: {data_path} ({len(df)} rows)")

    # ── 환경 생성 ──────────────────────────────────────────────────────────
    train_env = _make_env(df, leverage, mode="train")
    eval_env  = _make_env(df, leverage, mode="eval")

    # ── 모델 로드 (Baby 가중치 이식) ───────────────────────────────────────
    logger.info(f"  Baby 가중치 로드 중...")
    model = MaskablePPO.load(
        baby_model_path,
        env=train_env,
        tensorboard_log=log_dir,
        custom_objects={
            "learning_rate": FINETUNE_LR,
            "ent_coef":      FINETUNE_ENT_COEF,
        },
    )
    logger.info(f"  모델 로드 완료 (LR={FINETUNE_LR}, ent_coef={FINETUNE_ENT_COEF})")

    # ── 콜백 설정 ─────────────────────────────────────────────────────────
    eval_cb = CustomEvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=os.path.join(save_dir, "results"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    smart_stop = SmartStopCallback(
        eval_callback=eval_cb,
        patience=patience,
        eval_freq=eval_freq,
        reward_target=reward_target,
        entropy_threshold=0.01,
        total_timesteps=total_timesteps,
        no_improve_start_ratio=0.3,
    )

    # ── 학습 ──────────────────────────────────────────────────────────────
    start = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_cb, smart_stop],
            reset_num_timesteps=True,
            tb_log_name=tag,
        )
    except KeyboardInterrupt:
        logger.info("  [중단] 사용자 인터럽트 → 현재까지 학습된 모델 저장")

    # ── 최종 모델 저장 ────────────────────────────────────────────────────
    final_path = os.path.join(save_dir, f"final_model_{tag}.zip")
    model.save(final_path)
    elapsed = time.time() - start
    logger.info(f"  최종 모델 저장: {final_path}")
    logger.info(f"  소요 시간: {int(elapsed//60)}분 {int(elapsed%60)}초")

    return final_path


# ── CLI ───────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Baby→Full 커리큘럼 파인튜닝")
    p.add_argument("--baby-model-path", required=True,
                   help="Baby 환경 우승 모델 경로 (.zip)")
    p.add_argument("--leverage", type=int, default=2,
                   help="레버리지 배수 (default: 2)")
    p.add_argument("--tuning-profile", default="balanced",
                   choices=["stable", "balanced", "aggressive"],
                   help="튜닝 프로파일 (default: balanced)")
    p.add_argument("--data-path", required=True,
                   help="학습 데이터 CSV 경로")
    p.add_argument("--timesteps", type=int, default=500_000,
                   help="총 학습 스텝 수 (default: 500000)")
    p.add_argument("--model-dir", default=os.path.join(ROOT_DIR, "models", "finetuned"),
                   help="모델 저장 루트 디렉터리")
    p.add_argument("--log-dir", default=os.path.join(ROOT_DIR, "logs", "finetune"),
                   help="Tensorboard 로그 디렉터리")
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="평가 주기 (default: 10000)")
    p.add_argument("--n-eval-episodes", type=int, default=5,
                   help="평가 에피소드 수 (default: 5)")
    p.add_argument("--patience", type=int, default=10,
                   help="조기 종료 인내 횟수 (default: 10)")
    p.add_argument("--reward-target", type=float, default=5.0,
                   help="목표 보상 (달성 시 조기 종료, default: 5.0)")
    return p.parse_args()


if __name__ == "__main__":
    # ── 콜백 import 경로 fallback ─────────────────────────────────────────
    # 03_train_rl.py 에서 CustomEvalCallback 등을 직접 import
    try:
        from scripts.train_rl_components import (
            CustomEvalCallback, SmartStopCallback, PPO_TUNING_PROFILES,
        )
    except ImportError:
        # 03_train_rl.py 를 모듈로 직접 로드
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "train_rl", os.path.join(SCRIPT_DIR, "03_train_rl.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        CustomEvalCallback  = _mod.CustomEvalCallback   # noqa: F811
        SmartStopCallback   = _mod.SmartStopCallback    # noqa: F811
        PPO_TUNING_PROFILES = _mod.PPO_TUNING_PROFILES  # noqa: F811

    args = _parse_args()
    finetune(
        baby_model_path=args.baby_model_path,
        leverage=args.leverage,
        tuning_profile=args.tuning_profile,
        data_path=args.data_path,
        total_timesteps=args.timesteps,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        patience=args.patience,
        reward_target=args.reward_target,
    )
