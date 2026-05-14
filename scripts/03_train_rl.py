"""
사령관(RL Agent) 훈련 스크립트

단순화 원칙:
  - --count 만큼 반복 (1회 = 1 seed = 1 모델)
  - seed 는 내부에서 0~99999 랜덤 생성 (재현성보다 다양성 우선)
  - --tuning-profile 은 1개만 지정 (multi-profile 은 04_train_rl_batch 가 담당)
  - 생성된 tag 목록을 tags.txt 파일로 저장 (백테스트 입력용)
"""

import os
import sys
import random
import time
import argparse
import gc
import logging

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ── 경로 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

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
logger = logging.getLogger("Commander.Train")

_custom_log_dir = os.environ.get("CUSTOM_LOG_DIR")
if _custom_log_dir:
    os.makedirs(_custom_log_dir, exist_ok=True)
    logging.getLogger().addHandler(
        logging.FileHandler(os.path.join(_custom_log_dir, "train.log"), encoding='utf-8')
    )

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.envs.trading_env_baby import BabyLeverageTradingEnv as LeverageTradingEnv
from src.utils.platform_utils import configure_torch

configure_torch("cpu")


# ── 하이퍼파라미터 프로파일 ────────────────────────────────────────────────
PPO_TUNING_PROFILES = {
    "stable": {
        "policy_kwargs": dict(net_arch=[256, 256, 128]),
        "learning_rate": 1e-4,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "n_steps": 4096,
        "batch_size": 128,
    },
    "balanced": {
        "policy_kwargs": dict(net_arch=[256, 256, 128]),
        "learning_rate": 2e-4,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "n_steps": 2048,
        "batch_size": 64,
    },
    "aggressive": {
        "policy_kwargs": dict(net_arch=[256, 256, 128]),
        "learning_rate": 3e-4,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "n_steps": 2048,
        "batch_size": 64,
    },
}


# ── SmartStopCallback ──────────────────────────────────────────────────────
class SmartStopCallback(BaseCallback):
    """평가 보상 정체/정책 퇴화를 감지해 학습을 조기 종료합니다."""

    def __init__(self, eval_callback, patience=30, eval_freq=10_000,
                 entropy_threshold=-0.01, reward_target=None,
                 total_timesteps=5_000_000, no_improve_start_ratio=0.1,
                 verbose=1):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.patience = patience
        self.eval_freq = eval_freq
        self.reward_target = reward_target
        self.entropy_threshold = entropy_threshold
        ratio = min(1.0, max(0.1, float(no_improve_start_ratio)))
        self.no_improve_check_start_step = max(1, int(total_timesteps * ratio))
        self._no_improve_count = 0
        self._best_reward = -np.inf

    def _on_step(self) -> bool:
        entropy = self.logger.name_to_value.get("train/entropy_loss", None)
        if entropy is not None and entropy > self.entropy_threshold:
            logger.info(f"[SmartStop] 정책 퇴화 감지: entropy_loss={entropy:.6f} → 종료")
            return False

        if self.n_calls % self.eval_freq != 0:
            return True

        current_best = self.eval_callback.best_mean_reward
        if current_best == -np.inf:
            return True

        if current_best > self._best_reward:
            self._best_reward = current_best
            self._no_improve_count = 0
        else:
            if self.n_calls < self.no_improve_check_start_step:
                return True
            self._no_improve_count += 1

        if self._no_improve_count >= self.patience:
            logger.info(f"[SmartStop] Early Stopping (best={self._best_reward:.2f})")
            return False

        if self.reward_target is not None and self._best_reward >= self.reward_target:
            logger.info(f"[SmartStop] 목표 달성! eval reward={self._best_reward:.2f}")
            return False

        return True


# ── 단일 모델 훈련 ─────────────────────────────────────────────────────────
def train_one(seed, model_tag, leverage, tuning_profile, load_model_path,
              data_path, model_dir, log_dir,
              total_timesteps, eval_freq, patience, reward_target,
              entropy_threshold, no_improve_start_ratio,
              mutation_scale=1.0, n_envs=1):
    """seed 1개에 대한 PPO 훈련을 수행하고 저장합니다."""
    start = time.time()
    logger.info(f"  ▶ [{model_tag}] seed={seed} | lev={leverage}x | profile={tuning_profile} | n_envs={n_envs}")

    # hp는 항상 복사본 사용 (프로파일 원본 변경 방지)
    hp = dict(PPO_TUNING_PROFILES[tuning_profile])

    if n_envs > 1:
        # DummyVecEnv: 동일 프로세스 내 N개 환경을 배치 처리
        # 롤아웃 inference가 batch=N으로 묶여 Python 오버헤드 대폭 감소
        def _make_train_env():
            return Monitor(LeverageTradingEnv(data_path=data_path, leverage=leverage, mode="train"))
        train_env = DummyVecEnv([_make_train_env] * n_envs)
        # n_steps를 n_envs로 나눠 유효 롤아웃 버퍼 크기를 동일하게 유지
        hp["n_steps"] = max(64, hp["n_steps"] // n_envs)
        logger.info(f"    DummyVecEnv: n_envs={n_envs}, n_steps(per env)={hp['n_steps']}")
    else:
        train_env = Monitor(LeverageTradingEnv(data_path=data_path, leverage=leverage, mode="train"))

    eval_env = Monitor(LeverageTradingEnv(data_path=data_path, leverage=leverage, mode="eval"))

    if load_model_path and os.path.exists(load_model_path):
        logger.info(f"    부모 모델 로드: {load_model_path}")
        # 로컬 RNG 사용 → 전역 NumPy RNG 오염 방지
        rng = np.random.default_rng(seed)
        s = float(np.clip(mutation_scale, 0.0, 1.0))  # 적응형 변이 폭 스케일 (0.0~1.0)

        # 파인튜닝: 변이 폭 50% 완화 (Catastrophic Forgetting 방지, 기존 학습 보존)
        s_original = s
        s = s * 0.5
        logger.info(f"    [파인튜닝 모드] 변이 폭 완화: {s_original:.2f} → {s:.2f}")

        # 1. 엔트로피(ENT): [1-0.2s, 1+0.5s] 범위 변이, 클램프 [0.003, 0.03]
        mutated_ent = float(np.clip(
            hp["ent_coef"] * rng.uniform(1.0 - 0.2 * s, 1.0 + 0.5 * s),
            0.003, 0.03
        ))

        # 2. 학습률(LR): [1-0.2s, 1+0.2s] 범위 변이, 클램프 [1e-5, 5e-4]
        mutated_lr = float(np.clip(
            hp["learning_rate"] * rng.uniform(1.0 - 0.2 * s, 1.0 + 0.2 * s),
            1e-5, 5e-4
        ))

        # 3. vf_coef: ±15%*s 변이, 클램프 [0.3, 0.7]
        mutated_vf = float(np.clip(
            hp["vf_coef"] * rng.uniform(1.0 - 0.15 * s, 1.0 + 0.15 * s),
            0.3, 0.7
        ))

        # 4. clip_range: ±10%*s 변이, 클램프 [0.1, 0.3]
        mutated_clip = float(np.clip(
            0.2 * rng.uniform(1.0 - 0.10 * s, 1.0 + 0.10 * s),
            0.1, 0.3
        ))

        logger.info(
            f"    변이 적용(완화scale={s:.2f}) → "
            f"ent={mutated_ent:.5f}, lr={mutated_lr:.2e}, "
            f"vf={mutated_vf:.3f}, clip={mutated_clip:.3f}"
        )
        model = PPO.load(
            load_model_path, env=train_env, seed=seed,
            tensorboard_log=log_dir,
            custom_objects={
                "ent_coef":      mutated_ent,
                "learning_rate": mutated_lr,
                "vf_coef":       mutated_vf,
                "clip_range":    mutated_clip,
                "n_steps":       hp["n_steps"],  # n_envs 조정 반영
            }
        )

        # 5. 가우시안 노이즈: 정책 가중치에 상대적 미세 교란 (σ_rel = 0.003*s)
        noise_std = 0.003 * s
        if noise_std > 0:
            with torch.no_grad():
                for param in model.policy.parameters():
                    noise = torch.randn_like(param) * noise_std * param.abs().mean().clamp(min=1e-8)
                    param.add_(noise)
            logger.info(f"    가우시안 노이즈 적용 (σ_rel={noise_std:.4f})")
    else:
        model = PPO("MlpPolicy", train_env, verbose=0, seed=seed,
                    tensorboard_log=log_dir, **hp)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, model_tag),
        log_path=os.path.join(model_dir, model_tag, "results"),
        eval_freq=eval_freq, deterministic=True, render=False,
    )
    smart_stop = SmartStopCallback(
        eval_callback=eval_cb, patience=patience, eval_freq=eval_freq,
        reward_target=reward_target, entropy_threshold=entropy_threshold,
        total_timesteps=total_timesteps,
        no_improve_start_ratio=no_improve_start_ratio,
    )

    try:
        model.learn(total_timesteps=total_timesteps,
                    callback=[eval_cb, smart_stop],
                    reset_num_timesteps=False,
                    tb_log_name=model_tag)
    except KeyboardInterrupt:
        logger.warning("사용자 중단.")
    finally:
        os.makedirs(os.path.join(model_dir, model_tag), exist_ok=True)
        model.save(os.path.join(model_dir, model_tag, f"final_model_{model_tag}.zip"))
        del model, train_env, eval_env
        gc.collect()

    elapsed = int(time.time() - start)
    logger.info(f"  ✅ [{model_tag}] 완료 ({elapsed//60}분 {elapsed%60}초)")
    return model_tag


# ── 배치 실행 ──────────────────────────────────────────────────────────────
def _make_tag(model_dir, leverage, tuning_profile):
    """lev{N}_{prof}_seed{S}_{idx:03d} 형식의 순차 태그 생성."""
    os.makedirs(model_dir, exist_ok=True)
    prof_code = {"stable": "stb", "balanced": "bal", "aggressive": "agg"}.get(tuning_profile, "unk")
    seed = random.randint(0, 99999)
    prefix = f"lev{leverage}_{prof_code}_seed{seed}"
    # 혹시 같은 seed가 나왔을 때 충돌 방지용 suffix
    idx = 1
    for folder in os.listdir(model_dir):
        if folder.startswith(prefix + "_"):
            suffix = folder[len(prefix) + 1:]
            if suffix.isdigit():
                idx = max(idx, int(suffix) + 1)
    return f"{prefix}_{idx:03d}", seed


def run_train_batch(args):
    tags_created = []
    batch_start = time.time()

    for i in range(args.count):
        tag, seed = _make_tag(args.model_dir, args.leverage, args.tuning_profile)
        train_one(
            seed=seed, model_tag=tag, leverage=args.leverage,
            tuning_profile=args.tuning_profile,
            load_model_path=args.load_model,
            data_path=args.data_path,
            model_dir=args.model_dir, log_dir=args.log_dir,
            total_timesteps=args.timesteps, eval_freq=args.eval_freq,
            patience=args.patience, reward_target=args.reward_target,
            entropy_threshold=args.entropy_threshold,
            no_improve_start_ratio=args.no_improve_start_ratio,
            mutation_scale=args.mutation_scale,
            n_envs=args.n_envs,
        )
        tags_created.append(tag)
        total_elapsed = int(time.time() - batch_start)
        logger.info(f"🔄 [{i+1:03d}/{args.count:03d}] 누적 소요: {total_elapsed//60}분")

    # ── 생성된 태그 목록 파일 저장 ──────────────────────────────────────────
    tags_file = os.path.join(args.model_dir, "tags.txt")
    # 기존 목록에 누적 (같은 세대 폴더에 여러 batch 실행 시 병합)
    existing = []
    if os.path.exists(tags_file):
        with open(tags_file, "r", encoding="utf-8") as f:
            existing = [line.strip() for line in f if line.strip()]
    merged = sorted(set(existing + tags_created))
    with open(tags_file, "w", encoding="utf-8") as f:
        f.write("\n".join(merged) + "\n")
    logger.info(f"📄 태그 목록 저장: {tags_file} ({len(merged)}개)")

    return tags_created


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commander RL 단일 프로파일 훈련")

    parser.add_argument("--count",    type=int, default=1,
                        help="훈련할 모델 수 (각각 랜덤 seed)")
    parser.add_argument("--leverage", type=int, default=2,
                        help="레버리지 배수 (기본 2)")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                        help="총 훈련 타임스텝")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="평가 주기 (steps)")
    parser.add_argument("--tuning-profile",
                        choices=["stable", "balanced", "aggressive"],
                        default="balanced",
                        help="하이퍼파라미터 프로파일 (1개만 지정)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="파인튜닝용 부모 모델 (.zip)")
    parser.add_argument("--patience", type=int, default=30,
                        help="SmartStop 인내심 (평가 주기 단위)")
    parser.add_argument("--reward-target", type=float, default=1e8)
    parser.add_argument("--entropy-threshold", type=float, default=-0.01)
    parser.add_argument("--no-improve-start-ratio", type=float, default=0.1)
    parser.add_argument("--mutation-scale", type=float, default=1.0,
                        help="변이 폭 스케일 (1.0=최대, 0.0=변이 없음; run_evolution.py가 세대별 자동 조정)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="DummyVecEnv 병렬 환경 수 (기본=4). 1=단일 환경.\n"
                             "n_steps를 n_envs로 나눠 유효 버퍼 크기를 유지합니다.")

    default_data  = os.path.join(ROOT_DIR, "data", "signals", "base_signals_log.csv")
    default_model = os.path.join(ROOT_DIR, "checkpoints", "rl_generations")
    default_log   = os.path.join(ROOT_DIR, "logs", "train")

    parser.add_argument("--data-path",  type=str, default=default_data)
    parser.add_argument("--model-dir",  type=str, default=default_model)
    parser.add_argument("--log-dir",    type=str, default=default_log)

    args = parser.parse_args()

    # 환경변수 우선 (run_evolution.py / 04_train_rl_batch.py 연동)
    args.model_dir      = os.environ.get("CUSTOM_MODEL_DIR",  args.model_dir)
    args.log_dir        = os.environ.get("CUSTOM_LOG_DIR",    args.log_dir)
    args.mutation_scale = float(os.environ.get("MUTATION_SCALE", args.mutation_scale))

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"데이터 파일 없음: {args.data_path}")

    run_train_batch(args)
