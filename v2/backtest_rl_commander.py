import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from crypto_trading_env import CryptoTradingEnv
import warnings
warnings.filterwarnings('ignore')


def _resolve_model_path(model_path=None, model_tag=None):
    model_dir = os.path.join(BASE_DIR, "models", "rl_commander")
    if model_path is not None:
        if not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)
        return model_path

    if model_tag:
        tag = model_tag[:-4] if model_tag.endswith(".zip") else model_tag
        candidates_path = os.path.join(model_dir, "candidates", f"{tag}.zip")
        legacy_path = os.path.join(model_dir, f"{tag}.zip")
        if os.path.exists(candidates_path):
            return candidates_path
        if os.path.exists(legacy_path):
            return legacy_path
        return candidates_path

    return os.path.join(model_dir, "best_model.zip")


def run_rl_backtest(model_path=None, model_tag=None, output_suffix=""):
    print(f"\n{'='*50}")
    print(f"📈 3차 사령관 (RL Commander) 실전 백테스트 시작")
    print(f"{'='*50}")

    model_path = _resolve_model_path(model_path=model_path, model_tag=model_tag)

    data_path = os.path.join(BASE_DIR, "data", "base_signals_log.csv")

    if not os.path.exists(model_path):
        print(f"[ERROR] 모델 파일이 없습니다: {model_path}")
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
    entry_actions = []

    print("[INFO] 백테스트 시뮬레이션 가동 중...")
    
    while not done:
        # 사령관의 예측 (결정론적 행동 선택)
        action, _states = model.predict(obs, deterministic=True)
        act_val = int(action) if action.ndim == 0 else int(action[0])
        can_enter = (env.position == 0)
        
        # 🚨 핵심 패치 2: 스텝 진행 시 Gym(4개) vs Gymnasium(5개) 반환값 호환
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        balances.append(env.balance)

        # 실제 진입이 가능한 상태에서 발생한 진입 행동만 기록
        if can_enter and act_val in [1, 2]:
            entry_actions.append(act_val)

    # 3. 결과 요약
    final_balance = info['final_balance']
    pnl_pct = ((final_balance - env.initial_balance) / env.initial_balance) * 100
    total_trades = info['total_trades']
    win_rate = info.get('win_rate', 0.0)

    long_count = entry_actions.count(1)
    short_count = entry_actions.count(2)

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
    
    if not output_suffix and model_tag:
        output_suffix = model_tag

    if output_suffix:
        save_name = f"rl_backtest_result_{output_suffix}.png"
    else:
        save_name = "rl_backtest_result.png"
    save_fig_path = os.path.join(BASE_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    print(f"\n[INFO] 📈 결과 차트가 '{save_fig_path}' 파일로 저장되었습니다!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest RL commander with selected model")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Absolute path or v2-relative path of RL model zip")
    parser.add_argument("--model-tag", type=str, default=None,
                        help="Short tag in models/rl_commander/candidates, e.g. m001")
    parser.add_argument("--suffix", type=str, default="",
                        help="Output image suffix. e.g. seed42")
    args = parser.parse_args()

    run_rl_backtest(model_path=args.model_path,
                    model_tag=args.model_tag,
                    output_suffix=args.suffix)