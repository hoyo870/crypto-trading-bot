import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, initial_balance=10000.0, fee_rate=0.0005, mode='train'):
        super(CryptoTradingEnv, self).__init__()
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        
        print(f"[INFO] Trading Gym 데이터 로드 중 ({mode} 모드)...")
        self.df = pd.read_csv(data_path)
        
        split_idx = int(len(self.df) * 0.7)
        if mode == 'train':
            self.df = self.df.iloc[:split_idx].reset_index(drop=True)
        else:
            self.df = self.df.iloc[split_idx:].reset_index(drop=True)
            
        self.max_steps = len(self.df) - 1
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0 
        self.entry_price = 0.0
        self.total_trades = 0
        self.win_trades = 0
        # 최신 규격(튜플 반환)
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        current_price = row['close']
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            
        obs = np.array([
            row['long_score'],
            row['short_score'],
            row['context_score'],
            self.position,
            unrealized_pnl
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            self.balance *= (1 - self.fee_rate)
            
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
            self.balance *= (1 - self.fee_rate)
            
        elif action == 3 and self.position != 0:
            if self.position == 1:
                ret = (current_price - self.entry_price) / self.entry_price
            else:
                ret = (self.entry_price - current_price) / self.entry_price
                
            self.balance = self.balance * (1 + ret) * (1 - self.fee_rate)
            self.total_trades += 1
            
            # 🚨 핵심 패치 1: 진입/청산 수수료(총 0.1%)를 뺀 '진짜 순수익(Net Return)' 계산
            net_ret = ret - (self.fee_rate * 2)

            # 🚨 핵심 패치 2: 진짜 돈을 벌었을 때만 칭찬하고, 수수료 떼고 마이너스면 가차없이 페널티!
            if net_ret > 0:
                self.win_trades += 1
                reward = (net_ret * 100.0) + 2.0  # 승리 시 강력한 도파민(보상)
            else:
                reward = (net_ret * 100.0) - 1.0  # 패배 시 엄격한 전기충격(페널티)

            self.position = 0
            
        elif action == 0:
            if self.position == 0:
                # 🚨 핵심 패치 3: 관망 페널티를 살짝 줄입니다. (-0.0001)
                # 너무 쫄아서 무의미한 단타를 치지 않고, 진짜 기회를 기다릴 수 있게 해줍니다.
                reward = -0.0001 
            else:
                reward = 0.0

        self.current_step += 1
        
        if self.balance <= 0:
            terminated = True
        if self.current_step >= self.max_steps:
            truncated = True

        if terminated or truncated:
            info = {
                'final_balance': self.balance,
                'total_trades': self.total_trades,
                'win_rate': (self.win_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
            }

        return self._get_obs(), reward, terminated, truncated, info