import os
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
                 patience=10,
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

def train_commander():
    print(f"\n{'='*50}")
    print(f"🚀 3차 사령관 (RL Commander) 훈련 시작")
    print(f"{'='*50}")

    data_path = os.path.join(BASE_DIR, "data", "base_signals_log.csv")
    
    if not os.path.exists(data_path):
        print("[ERROR] base_signals_log.csv 파일이 없습니다. validate_base_signals.py를 먼저 실행하세요.")
        return

    # 환경 생성 (훈련용)
    env = CryptoTradingEnv(data_path=data_path, mode='train')
    
    # 평가 환경 생성 (가끔씩 시험을 보며 똑똑해지는지 체크)
    eval_env = CryptoTradingEnv(data_path=data_path, mode='test')

    # 사령관의 뇌(Policy) 구조: 은닉층 128, 64의 신경망
    policy_kwargs = dict(net_arch=[128, 64])
    
    # PPO 에이전트 생성
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                learning_rate=0.0003,
                ent_coef=0.01,        # 탐험 강제 → 정책 퇴화 방지
                vf_coef=0.5,          # 가치함수 학습 가중치
                n_steps=2048,
                batch_size=64)

    # 최고 성과를 낼 때마다 모델을 저장하는 콜백
    model_dir = os.path.join(BASE_DIR, "models", "rl_commander")
    os.makedirs(model_dir, exist_ok=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_dir,
                                 log_path=model_dir, eval_freq=10000,
                                 deterministic=True, render=False)

    smart_stop = SmartStopCallback(
        eval_callback=eval_callback,
        patience=10,              # eval 10회(=100,000 스텝) 연속 개선 없으면 종료
        eval_freq=10000,          # EvalCallback의 eval_freq와 반드시 동일하게
        entropy_threshold=-0.01,  # entropy_loss > -0.01 이면 퇴화로 판단
        reward_target=50.0,       # eval reward 50 이상이면 목표 달성으로 종료
    )

    # 훈련 (50만 번의 틱을 보면서 학습)
    print("[INFO] 강도 높은 실전 훈련에 돌입합니다...")
    model.learn(total_timesteps=3000000, callback=[eval_callback, smart_stop])

    print(f"\n🎉 훈련이 완료되었습니다! 최고 성능의 사령관이 {os.path.join(model_dir, 'best_model.zip')} 에 저장되었습니다.")

if __name__ == "__main__":
    train_commander()