import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from crypto_trading_env import CryptoTradingEnv
import warnings
warnings.filterwarnings('ignore')

def run_rl_backtest():
    print(f"\n{'='*50}")
    print(f"📈 3차 사령관 (RL Commander) 실전 백테스트 시작")
    print(f"{'='*50}")

    model_path = "models/rl_commander/best_model.zip"
    data_path = "data/base_signals_log.csv"

    if not os.path.exists(model_path):
        print("[ERROR] best_model.zip 파일이 없습니다. 훈련이 정상적으로 완료되었는지 확인하세요.")
        return

    # 1. 평가용 환경 로드 (Test 데이터 구간)
    env = CryptoTradingEnv(data_path=data_path, mode='test')
    
    # 2. 훈련된 사령관 로드
    print("[INFO] 최고 성능의 사령관 뇌를 이식 중...")
    model = PPO.load(model_path)

    # 🚨 핵심 패치 1: 리셋 시 튜플(Gymnasium) 반환 안전 처리
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    done = False
    
    # 기록용 리스트
    balances = [env.balance]
    actions_taken = []

    print("[INFO] 백테스트 시뮬레이션 가동 중...")
    
    while not done:
        # 사령관의 예측 (결정론적 행동 선택)
        action, _states = model.predict(obs, deterministic=True)
        
        # 🚨 핵심 패치 2: 스텝 진행 시 Gym(4개) vs Gymnasium(5개) 반환값 호환
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        balances.append(env.balance)
        
        # SB3 예측 결과가 배열일 수 있으므로 정수형(int) 스칼라로 변환
        act_val = int(action) if action.ndim == 0 else int(action[0])
        
        # 1: Long, 2: Short (의미 있는 행동만 기록)
        if act_val in [1, 2]:
            actions_taken.append(act_val)

    # 3. 결과 요약
    final_balance = info['final_balance']
    pnl_pct = ((final_balance - env.initial_balance) / env.initial_balance) * 100
    total_trades = info['total_trades']
    win_rate = info.get('win_rate', 0.0)

    long_count = actions_taken.count(1)
    short_count = actions_taken.count(2)

    print("\n========================================")
    print("📊 강화학습 사령관 백테스트 결과 리포트")
    print("========================================")
    print(f"초기 자본금 : {env.initial_balance:,.2f} USDT")
    print(f"최종 자본금 : {final_balance:,.2f} USDT")
    print(f"총 수익률   : {pnl_pct:>+8.2f}%")
    print(f"총 거래횟수 : {total_trades}회 (Long: {long_count} / Short: {short_count})")
    print(f"승률        : {win_rate:.2f}%")
    print("========================================")

    # 4. 자산 곡선(Equity Curve) 시각화 및 저장
    plt.figure(figsize=(12, 6))
    plt.plot(balances, label='RL Commander Balance', color='blue', linewidth=1.5)
    plt.axhline(env.initial_balance, color='red', linestyle='--', label='Initial Balance')
    
    plt.title('RL Commander Equity Curve (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps (5m Ticks)', fontsize=12)
    plt.ylabel('Balance (USDT)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_fig_path = "rl_backtest_result.png"
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    print(f"\n[INFO] 📈 결과 차트가 '{save_fig_path}' 파일로 저장되었습니다!")

if __name__ == "__main__":
    run_rl_backtest()