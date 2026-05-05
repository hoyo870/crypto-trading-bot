import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from crypto_trading_env import CryptoTradingEnv

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
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=0.0003)

    # 최고 성과를 낼 때마다 모델을 저장하는 콜백
    model_dir = os.path.join(BASE_DIR, "models", "rl_commander")
    os.makedirs(model_dir, exist_ok=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_dir,
                                 log_path=model_dir, eval_freq=10000,
                                 deterministic=True, render=False)

    # 훈련 (10만 번의 틱을 보면서 학습)
    print("[INFO] 강도 높은 실전 훈련에 돌입합니다...")
    model.learn(total_timesteps=1000000, callback=eval_callback)

    print(f"\n🎉 훈련이 완료되었습니다! 최고 성능의 사령관이 {os.path.join(model_dir, 'best_model.zip')} 에 저장되었습니다.")

if __name__ == "__main__":
    train_commander()