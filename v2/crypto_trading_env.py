import numpy as np
import pandas as pd
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import warnings
warnings.filterwarnings('ignore')

class CryptoTradingEnv(gym.Env):
    """
    3차 모델(사령관)을 훈련시키기 위한 커스텀 가상 매매 체육관.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, initial_balance=10000.0, fee_rate=0.0005, mode='train'):
        super(CryptoTradingEnv, self).__init__()
        
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        
        # 1. 1,2차 모델이 뽑아둔 점수 로그 데이터 로드
        print(f"[INFO] Trading Gym 데이터 로드 중 ({mode} 모드)...")
        self.df = pd.read_csv(data_path)
        
        # Train / Test 분리 (앞 70% Train, 뒤 30% Test)
        split_idx = int(len(self.df) * 0.7)
        if mode == 'train':
            self.df = self.df.iloc[:split_idx].reset_index(drop=True)
        else:
            self.df = self.df.iloc[split_idx:].reset_index(drop=True)
            
        self.max_steps = len(self.df) - 1

        # 2. 강화학습 환경 설정
        # 행동(Action) 공간: 0(관망), 1(롱 진입), 2(숏 진입), 3(청산)
        self.action_space = spaces.Discrete(4)
        
        # 상태(Observation) 공간: [Long점수, Short점수, Context점수, 현재포지션상태, 현재수익률]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        if hasattr(super(), 'reset'):
            try:
                super().reset(seed=seed)
            except TypeError:
                super().reset()
        self.balance = self.initial_balance
        self.current_step = 0
        
        # 포지션 상태: 0(없음), 1(롱 보유), -1(숏 보유)
        self.position = 0 
        self.entry_price = 0.0
        
        # 에피소드 평가를 위한 변수
        self.total_trades = 0
        self.win_trades = 0
        
        obs = self._get_obs()
        return (obs, {})

    def _get_obs(self):
        # 현재 틱의 데이터
        row = self.df.iloc[self.current_step]
        
        # 포지션별 현재 수익률 계산
        current_price = row['close']
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            
        # 상태 배열 반환
        obs = np.array([
            row['long_score'],
            row['short_score'],
            row['context_score'],
            self.position,        # 현재 포지션 여부
            unrealized_pnl        # 현재 포지션의 임시 수익률
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # 이전 잔고 기억 (수익 계산용)
        prev_balance = self.balance

        # ── 3. 사령관의 행동(Action) 처리 및 보상(Reward) ──
        if action == 1 and self.position == 0:
            # 롱 진입
            self.position = 1
            self.entry_price = current_price
            self.balance *= (1 - self.fee_rate)
            
        elif action == 2 and self.position == 0:
            # 숏 진입
            self.position = -1
            self.entry_price = current_price
            self.balance *= (1 - self.fee_rate)
            
        elif action == 3 and self.position != 0:
            # 청산(Close)
            if self.position == 1:
                ret = (current_price - self.entry_price) / self.entry_price
            else: # 숏 청산
                ret = (self.entry_price - current_price) / self.entry_price
                
            # 청산 후 원금 반영
            self.balance = self.balance * (1 + ret) * (1 - self.fee_rate)
            
            # 매매 횟수 및 승률 카운트
            self.total_trades += 1
            if ret > 0:
                self.win_trades += 1

            # 💰 야수형(Alpha) 보상 정책: 크게 먹었을 때 엄청난 칭찬!
            if ret > 0.015:  # 1.5% 이상 빅수익
                reward = 5.0
            elif ret > 0:
                reward = 1.0
            else:
                reward = -1.0 # 손절 페널티
                
            # 포지션 초기화
            self.position = 0
            
        elif action == 0:
            # 관망(Hold)
            if self.position == 0:
                # 무포지션인데 관망 잘했음 (기본 생존 점수)
                reward = 0.01
            else:
                # 포지션 보유 중인데 아직 청산 안 함
                reward = 0.0

        # 스텝 진행
        self.current_step += 1
        
        # 에피소드 종료 조건 (파산 혹은 데이터 끝)
        if self.balance <= 0 or self.current_step >= self.max_steps:
            if self.position != 0:
                if self.position == 1:
                    ret = (current_price - self.entry_price) / self.entry_price
                else:
                    ret = (self.entry_price - current_price) / self.entry_price
                self.balance = self.balance * (1 + ret) * (1 - self.fee_rate)
                self.total_trades += 1
                if ret > 0:
                    self.win_trades += 1
                self.position = 0

            terminated = self.balance <= 0
            truncated = self.current_step >= self.max_steps
            info = {
                'final_balance': self.balance,
                'total_trades': self.total_trades,
                'win_rate': (self.win_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
            }

        return self._get_obs(), reward, terminated, truncated, info