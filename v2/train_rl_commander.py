import argparse
import os
import re
import shutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from crypto_trading_env import CryptoTradingEnv


class SmartStopCallback(BaseCallback):
    """
    3가지 조건으로 자동 학습 종료:
    1. Early Stopping : eval reward가 patience 횟수(eval 단위) 연속 개선 없으면 종료
    2. 정책 퇴화 감지 : entropy_loss > entropy_threshold 이면 종료 (0에 수렴 = 퇴화)
    3. 목표 달성     : eval reward >= reward_target 이면 종료
    """
    def __init__(self, eval_callback,
                 patience=20,
                 eval_freq=10000,
                 entropy_threshold=-0.01,
                 reward_target=50.0,
                 verbose=1):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.patience = patience
        self.eval_freq = eval_freq          # EvalCallback과 동일하게 맞춰야 함
        self.entropy_threshold = entropy_threshold
        self.reward_target = reward_target
        self._no_improve_count = 0
        self._best_reward = -np.inf

    def _on_step(self) -> bool:
        # ── 3. 정책 퇴화 감지 (매 스텝 체크 — 즉각 반응 필요) ──────────
        entropy = self.logger.name_to_value.get("train/entropy_loss", None)
        if entropy is not None and entropy > self.entropy_threshold:
            if self.verbose:
                print(f"\n[SmartStop] 💀 정책 퇴화 감지: entropy_loss={entropy:.6f} "
                      f"> {self.entropy_threshold} → 종료")
            return False

        # ── eval_freq 스텝마다만 patience/목표 체크 ──────────────────────
        # n_calls는 1부터 시작하므로, eval_freq 배수일 때만 평가
        if self.n_calls % self.eval_freq != 0:
            return True

        current_best = self.eval_callback.best_mean_reward

        # EvalCallback이 아직 한 번도 평가 안 했으면 스킵
        if current_best == -np.inf:
            return True

        # ── 1. Early Stopping ────────────────────────────────────────────
        if current_best > self._best_reward:
            self._best_reward = current_best
            self._no_improve_count = 0
            if self.verbose:
                print(f"[SmartStop] ✅ 개선됨: best_reward={self._best_reward:.2f}")
        else:
            self._no_improve_count += 1
            if self.verbose:
                print(f"[SmartStop] ⚠️  개선 없음 {self._no_improve_count}/{self.patience} "
                      f"(best={self._best_reward:.2f})")

        if self._no_improve_count >= self.patience:
            if self.verbose:
                print(f"\n[SmartStop] ⏹  Early Stopping: {self.patience}회 연속 개선 없음 "
                      f"(best={self._best_reward:.2f})")
            return False

        # ── 2. 목표 달성 ─────────────────────────────────────────────────
        if self._best_reward >= self.reward_target:
            if self.verbose:
                print(f"\n[SmartStop] 🎯 목표 달성! eval reward={self._best_reward:.2f} "
                      f">= {self.reward_target}")
            return False

        return True  # 계속 훈련

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_regime_eval_starts(max_steps, eval_window=20_000):
    # 상승/하락/횡보가 섞이도록 전체 기간을 4개 구간으로 분할해 고정 시작점 사용
    anchors = [0.0, 0.25, 0.50, 0.75]
    max_start = max(0, max_steps - eval_window)
    starts = [min(max_start, int(max_steps * q)) for q in anchors]
    starts = sorted(set(starts))
    return starts or [0]


class RegimeEvalEnv(CryptoTradingEnv):
    def __init__(self, data_path, eval_starts, eval_window=20_000):
        super().__init__(data_path=data_path)
        self.eval_starts = list(eval_starts)
        self.eval_window = eval_window
        self.eval_idx = 0

    def reset(self, seed=None, options=None):
        options = dict(options or {})
        if 'start_step' not in options:
            options['start_step'] = self.eval_starts[self.eval_idx % len(self.eval_starts)]
        if 'max_ep_steps' not in options:
            options['max_ep_steps'] = self.eval_window
        self.eval_idx += 1
        return super().reset(seed=seed, options=options)


def _normalize_tag(tag):
    clean = re.sub(r"[^a-zA-Z0-9_-]", "", str(tag)).lower()
    return clean or None


def _next_model_tag(candidates_dir):
    max_idx = 0
    if os.path.isdir(candidates_dir):
        for name in os.listdir(candidates_dir):
            m = re.fullmatch(r"m(\d{3})\.zip", name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return f"m{max_idx + 1:03d}"

def train_commander(total_timesteps=3000000,
                    eval_freq=10000,
                    patience=10,
                    reward_target=None,
                    entropy_threshold=-0.01,
                    seed=42,
                    model_tag=None):
    print(f"\n{'='*50}")
    print(f"🚀 3차 사령관 (RL Commander) 훈련 시작")
    print(f"{'='*50}")

    data_path = os.path.join(BASE_DIR, "data", "base_signals_log.csv")
    
    if not os.path.exists(data_path):
        print("[ERROR] base_signals_log.csv 파일이 없습니다. validate_base_signals.py를 먼저 실행하세요.")
        return

    # 환경 생성 (훈련용: 랜덤 시작점으로 전체 기간 탐색)
    env = CryptoTradingEnv(data_path=data_path)

    # 평가 환경 생성 (시장 국면별 고정 시작점으로 일반화 성능 측정)
    eval_starts = _build_regime_eval_starts(max_steps=env.max_steps, eval_window=20_000)
    eval_env = RegimeEvalEnv(data_path=data_path, eval_starts=eval_starts, eval_window=20_000)
    print(f"[INFO] 평가 시작점(국면 분할): {eval_starts}")

    # 사령관의 뇌(Policy) 구조: 은닉층 128, 64의 신경망
    policy_kwargs = dict(net_arch=[128, 64])
    
    # PPO 에이전트 생성
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                learning_rate=0.0003,
                ent_coef=0.01,        # 탐험 강제 → 정책 퇴화 방지
                vf_coef=0.5,          # 가치함수 학습 가중치
                n_steps=2048,
                batch_size=64,
                seed=seed)

    # 저장 경로: runs/<tag>/best_model.zip + candidates/<tag>.zip
    model_dir = os.path.join(BASE_DIR, "models", "rl_commander")
    runs_dir = os.path.join(model_dir, "runs")
    candidates_dir = os.path.join(model_dir, "candidates")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(candidates_dir, exist_ok=True)

    requested_tag = _normalize_tag(model_tag) if model_tag else None
    if model_tag and requested_tag is None:
        requested_tag = _next_model_tag(candidates_dir)
        print(f"[WARN] 유효하지 않은 --tag 입니다. 자동 태그로 대체: {requested_tag}")

    final_tag = requested_tag or _next_model_tag(candidates_dir)
    run_dir = os.path.join(runs_dir, final_tag)
    os.makedirs(run_dir, exist_ok=True)

    eval_callback = EvalCallback(eval_env, best_model_save_path=run_dir,
                                 log_path=run_dir, eval_freq=eval_freq,
                                 deterministic=True, render=False,
                                 n_eval_episodes=len(eval_starts))

    smart_stop = SmartStopCallback(
        eval_callback=eval_callback,
        patience=patience,            # eval 연속 개선 없으면 종료
        eval_freq=eval_freq,          # EvalCallback의 eval_freq와 반드시 동일하게
        entropy_threshold=entropy_threshold,
        reward_target=(float("inf") if reward_target is None else reward_target),
    )

    # 훈련
    print("[INFO] 강도 높은 실전 훈련에 돌입합니다...")
    print(f"[INFO] 모델 태그: {final_tag}")
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, smart_stop])

    best_model_in_run = os.path.join(run_dir, "best_model.zip")
    if os.path.exists(best_model_in_run):
        promoted_path = os.path.join(candidates_dir, f"{final_tag}.zip")
        shutil.copy2(best_model_in_run, promoted_path)
        shutil.copy2(best_model_in_run, os.path.join(model_dir, "best_model.zip"))
        print(f"[INFO] 후보 모델 저장: {promoted_path}")
        print(f"[INFO] 기본 모델 갱신: {os.path.join(model_dir, 'best_model.zip')}")

    print(f"\n🎉 훈련 완료. 결과 폴더: {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Commander with smart stop")
    parser.add_argument("--total-timesteps", type=int, default=3000000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--entropy-threshold", type=float, default=-0.01)
    parser.add_argument("--reward-target", type=float, default=50.0,
                        help="Set to a high value (e.g. 1e9) to effectively disable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=None,
                        help="Short model tag, e.g. m007 or s42a (default: auto mNNN)")
    args = parser.parse_args()

    reward_target = args.reward_target
    if reward_target >= 1e8:
        reward_target = None

    train_commander(total_timesteps=args.total_timesteps,
                    eval_freq=args.eval_freq,
                    patience=args.patience,
                    reward_target=reward_target,
                    entropy_threshold=args.entropy_threshold,
                    seed=args.seed,
                    model_tag=args.tag)