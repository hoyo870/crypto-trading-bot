"""
사령관(RL Agent) 훈련 스크립트 (단일 및 배치 통합본)

기존 train_rl_commander.py 와 run_train.py 를 하나로 통합하여,
단일 시드 훈련부터 N개의 시드 일괄 배치 훈련까지 모두 이 스크립트 하나로 제어합니다.
"""

import os
import sys
import re
import time
import argparse
import random
import gc
from datetime import datetime
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

# ── 경로 설정 (새로운 아키텍처 반영) ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # scripts/
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                       # 프로젝트 루트
SRC_DIR = os.path.join(ROOT_DIR, "src")                     # 소스 코드 디렉토리

# 환경(Env) 및 모델(Models) 임포트를 위해 src 경로 추가
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.envs.trading_env_baby import BabyLeverageTradingEnv as LeverageTradingEnv


# ── 하이퍼파라미터 프로파일 ──────────────────────────────────────────────────
PPO_TUNING_PROFILES = {
    "stable": {
        "policy_kwargs": dict(net_arch=[256, 256, 128]),
        "learning_rate": 1e-4,
        "ent_coef": 0.01,  # 탐험심 강제 주입
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

# ── 콜백 (스마트 조기 종료) ────────────────────────────────────────────────
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
        self.no_improve_start_ratio = ratio
        self.no_improve_check_start_step = max(1, int(total_timesteps * ratio))
        self._no_improve_count = 0
        self._best_reward = -np.inf

    def _on_step(self) -> bool:
        entropy = self.logger.name_to_value.get("train/entropy_loss", None)
        if entropy is not None and entropy > self.entropy_threshold:
            if self.verbose:
                print(f"\n[SmartStop] 정책 퇴화 감지: entropy_loss={entropy:.6f} → 종료")
            return False

        if self.n_calls % self.eval_freq != 0:
            return True

        current_best = self.eval_callback.best_mean_reward
        if current_best == -np.inf:
            return True

        if current_best > self._best_reward:
            self._best_reward = current_best
            self._no_improve_count = 0
            if self.verbose:
                print(f"[SmartStop] 개선됨: best_reward={self._best_reward:.2f}")
        else:
            if self.n_calls < self.no_improve_check_start_step:
                if self.verbose:
                    print(f"[SmartStop] 워밍업 구간: no-improve 체크 보류 ({self.n_calls}/{self.no_improve_check_start_step} steps)")
                return True
            self._no_improve_count += 1
            if self.verbose:
                print(f"[SmartStop] 개선 없음 {self._no_improve_count}/{self.patience} (best={self._best_reward:.2f})")

        if self._no_improve_count >= self.patience:
            if self.verbose:
                print(f"\n[SmartStop] Early Stopping (best={self._best_reward:.2f})")
            return False

        if self.reward_target is not None and self._best_reward >= self.reward_target:
            if self.verbose:
                print(f"\n[SmartStop] 목표 달성! eval reward={self._best_reward:.2f}")
                return False

        return True


# ── 코어 훈련 함수 ────────────────────────────────────────────────────────
def train_commander(
    total_timesteps, eval_freq, patience, reward_target, entropy_threshold,
    no_improve_start_ratio, seed, model_tag, leverage, load_model_path,
    improved_hp, split_mode, train_ratio, eval_ratio, train_ep_steps,
    eval_window, data_path, model_dir, log_dir, tuning_profile
):
    print(f"\\n{'='*60}")
    print(f"🚀 Commander RL 훈련 시작 (레버리지 {leverage}x, Seed: {seed})")
    print(f"{'='*60}")

    # 환경 생성 (Train / Eval) - 향후 holdout 분할 등을 env 내부에서 처리한다고 가정
    train_env = LeverageTradingEnv(data_path=data_path, leverage=leverage, mode="train")
    eval_env  = LeverageTradingEnv(data_path=data_path, leverage=leverage, mode="eval")

    train_env = Monitor(train_env)
    eval_env  = Monitor(eval_env)

    # 파라미터 로드
    hp = PPO_TUNING_PROFILES.get(tuning_profile, PPO_TUNING_PROFILES["balanced"])

    if load_model_path and os.path.exists(load_model_path):
        print(f"[INFO] 기존 부모 모델 로드 중: {load_model_path}")
        # 세대 진화 (Gen2, Gen3...) 시 탐험심 주입
        custom_objects = {"ent_coef": hp["ent_coef"], "learning_rate": hp["learning_rate"]}
        model = PPO.load(
            load_model_path,
            env=train_env,
            custom_objects=custom_objects,
            seed=seed,
            tensorboard_log=log_dir
        )
    else:
        print(f"[INFO] 백지 상태(Random Weights)에서 훈련 시작")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=seed,
            tensorboard_log=log_dir,
            **hp
        )

    # 콜백 설정
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, model_tag),
        log_path=os.path.join(model_dir, model_tag, "results"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    smart_stop = SmartStopCallback(
        eval_callback=eval_callback,
        patience=patience, 
        eval_freq=eval_freq,
        reward_target=reward_target,
        entropy_threshold=entropy_threshold,
        total_timesteps=total_timesteps,
        no_improve_start_ratio=no_improve_start_ratio,
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, smart_stop],
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\\n[INFO] 사용자에 의해 학습이 강제 중단되었습니다.")
    finally:
        # 최종 모델 저장 (가장 좋았던 모델은 EvalCallback이 이미 저장함)        os.makedirs(os.path.join(model_dir, model_tag), exist_ok=True)  # 디렉터리 보장        final_path = os.path.join(model_dir, model_tag, f"final_model_{model_tag}.zip")
        model.save(final_path)
        print(f"✅ 훈련 종료! 최종 모델 저장됨: {final_path}")
        
        # 메모리 누수 방지
        del model, train_env, eval_env
        gc.collect()


# ── 배치 실행 헬퍼 ─────────────────────────────────────────────────────────
def _next_tag(model_dir, leverage, seed):
    os.makedirs(model_dir, exist_ok=True)
    prefix = f"lev{int(leverage)}_seed{int(seed)}"
    max_idx = 0
    for folder in os.listdir(model_dir):
        if folder.startswith(prefix + "_"):
            suffix = folder[len(prefix) + 1:]
            if suffix.isdigit() and len(suffix) == 3:
                max_idx = max(max_idx, int(suffix))
    return f"{prefix}_{max_idx + 1:03d}"

def run_train_batch(args):
    # 시드 배열 생성
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(',')]
        if len(seeds) < args.count:
            raise ValueError(f"--seeds 개수({len(seeds)})가 --count({args.count})보다 작습니다.")
        seeds = seeds[:args.count]
    else:
        seeds = [args.base_seed + i * args.seed_step for i in range(args.count)]

    start_time = time.time()
    
    for i, seed in enumerate(seeds):
        print(f"\\n[BATCH] 진행 상황: {i+1} / {len(seeds)} (현재 시드: {seed})")
        
        # 저장 폴더 태그 생성
        if args.tag and len(seeds) == 1:
            tag = str(args.tag)
        elif args.tag and len(seeds) > 1:
            tag = f"{str(args.tag)}_{i + 1:03d}"
        else:
            tag = _next_tag(args.model_dir, args.leverage, seed)
        
        train_commander(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            patience=args.patience,
            reward_target=args.reward_target,
            entropy_threshold=args.entropy_threshold,
            no_improve_start_ratio=args.no_improve_start_ratio,
            seed=seed,
            model_tag=tag,
            leverage=args.leverage,
            load_model_path=args.load_model,
            improved_hp=args.improved_hp,
            split_mode=args.split_mode,
            train_ratio=args.train_ratio,
            eval_ratio=args.eval_ratio,
            train_ep_steps=args.train_ep_steps,
            eval_window=args.eval_window,
            data_path=args.data_path,
            model_dir=args.model_dir,
            log_dir=args.log_dir,
            tuning_profile=args.tuning_profile
        )

    if args.top_k > 0:
        print(f"[INFO] --top-k={args.top_k} 는 통합 스크립트에서 아직 미사용입니다. (호환 인자로만 수용)")

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\\n{'='*60}")
    print(f"🎉 일괄 훈련 배치가 모두 종료되었습니다!")
    print(f"⏱️ 총 소요 시간: {int(hours)}시간 {int(mins)}분 {int(secs)}초")
    print(f"{'='*60}\\n")


# ── 메인 진입점 (Argparse 통합) ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commander RL 훈련 (단일/배치 통합 스크립트)")
    
    # 1. 반복 및 시드 설정
    parser.add_argument("--count", type=int, default=1, help="실행할 총 훈련(시드) 개수")
    parser.add_argument("--seed", type=int, default=None, help="단일 훈련용 특정 시드 (입력 시 count 무시)")
    parser.add_argument("--base-seed", type=int, default=42, help="배치 훈련 시 시작 시드")
    parser.add_argument("--seed-step", type=int, default=100, help="배치 훈련 시 시드 증가폭")
    parser.add_argument("--seeds", type=str, default="", help="콤마로 구분된 지정 시드 목록 (예: 42,1042,2042)")
    
    # 2. 훈련 기본 설정
    parser.add_argument("--leverage", type=int, default=2, help="레버리지 배수 (기본 2)")
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="총 훈련 타임스텝 수")
    parser.add_argument("--total-timesteps", dest="timesteps", type=int,
                        help="--timesteps 별칭 (레거시 호환)")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="평가 및 최고 모델 저장 주기")
    parser.add_argument("--tuning-profile", type=str, choices=["stable", "balanced", "aggressive"], default="balanced", help="학습 프로파일 선택")
    parser.add_argument("--load-model", type=str, default=None, help="커리큘럼 학습용 부모 모델(.zip) 경로")
    
    # 3. 조기 종료(SmartStop) 콜백 설정
    parser.add_argument("--patience", type=int, default=30, help="성능 개선이 없을 때 기다릴 최대 주기")
    parser.add_argument("--reward-target", type=float, default=1e8, help="이 평균 보상에 도달하면 즉시 훈련 종료")
    parser.add_argument("--entropy-threshold", type=float, default=-0.01, help="entropy_loss 임계치")
    parser.add_argument("--no-improve-start-ratio", type=float, default=0.1, help="전체 타임스텝 중 조기 종료를 허용할 시작 비율 (0.1 = 10% 진행 후)")
    
    # 4. 데이터 및 저장 경로 (신규 아키텍처 기본값)
    default_data_path = os.path.join(ROOT_DIR, "data", "signals", "base_signals_log.csv")
    default_model_dir = os.path.join(ROOT_DIR, "checkpoints", "rl_generations")
    default_log_dir = os.path.join(ROOT_DIR, "logs", "train")
    
    parser.add_argument("--data-path", type=str, default=default_data_path, help="베이스 참모진 신호 데이터 경로")
    parser.add_argument("--model-dir", type=str, default=default_model_dir, help="모델 체크포인트 저장 폴더")
    parser.add_argument("--log-dir", type=str, default=default_log_dir, help="텐서보드 로그 폴더")
    parser.add_argument("--tag", type=str, default=None, help="모델 태그 override (count=1 권장)")
    parser.add_argument("--top-k", type=int, default=0, help="레거시 호환용 인자(현재 미사용)")
    
    # 5. 기타 레거시 옵션
    parser.add_argument("--improved-hp", action="store_true", help="(레거시) 개선된 하이퍼파라미터 사용")
    parser.add_argument("--split-mode", type=str, default="holdout", help="데이터 분할 방식")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train 데이터 비율")
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="Eval 데이터 비율 (호환 인자)")
    parser.add_argument("--train-ep-steps", type=int, default=20_000, help="학습 에피소드 길이 (호환 인자)")
    parser.add_argument("--eval-window", type=int, default=20_000, help="평가 에피소드 길이 (호환 인자)")

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"입력 데이터 파일이 없습니다: {args.data_path}")

    run_train_batch(args)