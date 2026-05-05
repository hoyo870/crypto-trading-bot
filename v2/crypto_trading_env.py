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
        df = pd.read_csv(data_path)
        
        split_idx = int(len(df) * 0.7)
        if mode == 'train':
            df = df.iloc[:split_idx].reset_index(drop=True)
        else:
            df = df.iloc[split_idx:].reset_index(drop=True)
            
        self.max_steps = len(df) - 1

        # 🚀 핵심 엔진 교체: Pandas DataFrame을 Numpy Array로 미리 변환해 메모리에 적재
        self.closes = df['close'].values
        self.long_scores = df['long_score'].values
        self.short_scores = df['short_score'].values
        self.context_scores = df['context_score'].values

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
        return self._get_obs(), {}

    def _get_obs(self):
        # Numpy 배열에서 직접 가져와 속도 극대화
        current_price = self.closes[self.current_step]
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            
        obs = np.array([
            self.long_scores[self.current_step],
            self.short_scores[self.current_step],
            self.context_scores[self.current_step],
            self.position,
            unrealized_pnl
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.closes[self.current_step]
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
            
            net_ret = ret - (self.fee_rate * 2)

            if net_ret > 0:
                self.win_trades += 1
                reward = (net_ret * 100.0) + 2.0  
            else:
                reward = (net_ret * 100.0) - 1.0  

            self.position = 0
            
        elif action == 0:
            if self.position == 0:
                reward = -0.0001  # 무포지션 관망 패널티
            else:
                reward = -0.0005  # 포지션 보유 중 hold → 기회비용 패널티 추가

        self.current_step += 1
        
        if self.balance <= 0:
            terminated = True
        if self.current_step >= self.max_steps:
            truncated = True

        if terminated or truncated:
            # 에피소드 종료 시 미청산 포지션 강제 청산 + 패널티
            if self.position != 0:
                final_price = self.df.iloc[min(self.current_step, self.max_steps - 1)]['close']
                if self.position == 1:
                    forced_ret = (final_price - self.entry_price) / self.entry_price
                else:
                    forced_ret = (self.entry_price - final_price) / self.entry_price
                net_ret = forced_ret - (self.fee_rate * 2)
                self.balance = self.balance * (1 + forced_ret) * (1 - self.fee_rate)
                self.total_trades += 1
                if net_ret > 0:
                    self.win_trades += 1
                # 강제 청산 추가 패널티: 스스로 청산하지 않은 것에 대한 벌점
                reward += -2.0
                self.position = 0

            info = {
                'final_balance': self.balance,
                'total_trades': self.total_trades,
                'win_rate': (self.win_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
            }

        return self._get_obs(), reward, terminated, truncated, info