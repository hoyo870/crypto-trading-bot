import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

# ── 하이퍼파라미터 ───────────────────────────────────────────────
MIN_HOLD_STEPS  = 12     # 최소 보유 시간: 1시간 (12스텝)
MAX_HOLD_STEPS  = 2016   # 강제 청산 한도: 1주 (7일 × 288스텝/일)
HOLD_BASE_STEPS = 288    # 보유 패널티 시작: 24시간
SHARPE_WINDOW   = 50     # Sharpe 계산용 최근 거래 수
DD_THRESHOLD    = 0.10   # 드로우다운 패널티 발동 기준: -10%
MIN_EP_STEPS    = 10_000 # 에피소드 최소 잔여 스텝 (랜덤 시작 상한)
MAX_EP_STEPS    = 20_000 # 에피소드 최대 길이 (약 70일): 보상 누적 폭발 방지
REWARD_CLIP     = 10.0   # per-step 보상 클리핑 범위
IMBALANCE_FREE_BAND = 0.60  # Long/Short 불균형 허용 구간
IMBALANCE_PENALTY_COEF = 0.30
TAIL_LOSS_THRESHOLD = 0.01  # 순손실 1% 초과분부터 꼬리손실 패널티
TAIL_LOSS_COEF = 0.20
# ────────────────────────────────────────────────────────────────


class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, initial_balance=10000.0, fee_rate=0.0005, mode=None):
        super().__init__()
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        # mode 파라미터는 하위호환성 보존용 스텀으로만 존재하며, 실제로는 사용되지 않음.
        # 항상 전체 데이터(315,056행)를 사용하며 reset() 시 시작점을 랜덤 배정함.

        print("[INFO] Trading Gym 데이터 로드 중 (전체 데이터 모드)...")
        df = pd.read_csv(data_path)
        # 항상 전체 데이터 기간 사용 (train/test split 없음)
        self.max_steps = len(df) - 1

        # ── 가격·시그널 배열 ──
        self.closes         = df['close'].values.astype(np.float32)
        self.long_scores    = df['long_score'].values.astype(np.float32)
        self.short_scores   = df['short_score'].values.astype(np.float32)
        self.context_scores = df['context_score'].values.astype(np.float32)

        # ── 시간 피처: sin/cos 인코딩으로 주기성 보존 ──
        dt = pd.to_datetime(df['datetime'])
        hours = dt.dt.hour.values
        dows  = dt.dt.dayofweek.values
        self.hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
        self.hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
        self.dow_sin  = np.sin(2 * np.pi * dows  /  7).astype(np.float32)
        self.dow_cos  = np.cos(2 * np.pi * dows  /  7).astype(np.float32)

        self.action_space = spaces.Discrete(4)
        # obs(9): long_score, short_score, context_score, position, unrealized_pnl,
        #         hour_sin, hour_cos, dow_sin, dow_cos
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )

    # ────────────────────── reset ──────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # options={'start_step': N} 으로 시작점 고정 가능 (백테스트용)
        # 미지정 시 랜덤 시작 → 전체 데이터 균등 탐색, 최소 MIN_EP_STEPS 잔여 보장
        if options is not None and 'start_step' in options:
            self.start_step = int(options['start_step'])
        else:
            max_start = max(0, self.max_steps - MIN_EP_STEPS)
            self.start_step = int(self.np_random.integers(0, max_start + 1))
        if options is not None and 'max_ep_steps' in options:
            max_ep_steps = options['max_ep_steps']
            self.max_episode_steps = None if max_ep_steps is None else int(max_ep_steps)
        else:
            self.max_episode_steps = MAX_EP_STEPS
        self.current_step = self.start_step

        self.balance      = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position     = 0
        self.entry_price  = 0.0
        self.entry_step   = 0
        self.total_trades = 0
        self.win_trades   = 0
        self.long_entries = 0
        self.short_entries = 0
        self.recent_returns = []   # Sharpe 계산 버퍼 (float 리스트)
        return self._get_obs(), {}

    # ────────────────────── obs ────────────────────────────────
    def _get_obs(self):
        i = self.current_step
        current_price = self.closes[i]
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price

        return np.array([
            self.long_scores[i],
            self.short_scores[i],
            self.context_scores[i],
            float(self.position),
            float(np.clip(unrealized_pnl, -1.0, 1.0)),
            self.hour_sin[i],
            self.hour_cos[i],
            self.dow_sin[i],
            self.dow_cos[i],
        ], dtype=np.float32)

    # ────────────────────── step ───────────────────────────────
    def step(self, action):
        i = self.current_step
        current_price = self.closes[i]
        reward    = 0.0
        terminated = False
        truncated  = False
        info       = {}

        hold_steps = (self.current_step - self.entry_step) if self.position != 0 else 0

        # ① 강제 청산: 1주 초과 보유 시 exit 강제 적용
        if self.position != 0 and hold_steps >= MAX_HOLD_STEPS:
            action = 3

        # 최소 보유 시간 전 청산은 무효 처리 (조기 청산 남용 방지)
        if action == 3 and self.position != 0 and hold_steps < MIN_HOLD_STEPS:
            action = 0
            reward -= 0.02

        # ── 행동 처리 ──
        if action == 1 and self.position == 0:          # Long 진입
            self.position    = 1
            self.entry_price = current_price
            self.entry_step  = self.current_step
            self.balance    *= (1 - self.fee_rate)
            self.long_entries += 1

            # Long/Short 행동 불균형이 과도하면 진입 시 페널티
            total_entries = self.long_entries + self.short_entries
            imbalance = abs(self.long_entries - self.short_entries) / max(1, total_entries)
            if total_entries >= 10 and imbalance > IMBALANCE_FREE_BAND:
                reward -= (imbalance - IMBALANCE_FREE_BAND) * IMBALANCE_PENALTY_COEF

        elif action == 2 and self.position == 0:        # Short 진입
            self.position    = -1
            self.entry_price = current_price
            self.entry_step  = self.current_step
            self.balance    *= (1 - self.fee_rate)
            self.short_entries += 1

            total_entries = self.long_entries + self.short_entries
            imbalance = abs(self.long_entries - self.short_entries) / max(1, total_entries)
            if total_entries >= 10 and imbalance > IMBALANCE_FREE_BAND:
                reward -= (imbalance - IMBALANCE_FREE_BAND) * IMBALANCE_PENALTY_COEF

        elif action == 3 and self.position != 0:        # 청산
            if self.position == 1:
                ret = (current_price - self.entry_price) / self.entry_price
            else:
                ret = (self.entry_price - current_price) / self.entry_price

            self.balance = self.balance * (1 + ret) * (1 - self.fee_rate)
            self.total_trades += 1
            net_ret = ret - (self.fee_rate * 2)

            # ② Sharpe 기반 보상: 변동성 대비 수익률 품질 평가
            self.recent_returns.append(net_ret)
            if len(self.recent_returns) > SHARPE_WINDOW:
                self.recent_returns.pop(0)
            if len(self.recent_returns) >= 2:
                sharpe_bonus = (net_ret / (np.std(self.recent_returns) + 1e-8)) * 0.5
            else:
                sharpe_bonus = 0.0

            if net_ret > 0:
                self.win_trades += 1
                base_reward = (net_ret * 100.0) + 1.0
                reward = base_reward + sharpe_bonus
            else:
                # 꼬리손실(큰 손실)에 비선형 패널티를 더해 손익비 중심으로 유도
                tail_loss = max(0.0, abs(net_ret) - TAIL_LOSS_THRESHOLD)
                tail_penalty = ((tail_loss * 100.0) ** 2) * TAIL_LOSS_COEF
                reward = (net_ret * 100.0) - 1.5 + sharpe_bonus - tail_penalty

            self.position = 0
            self.peak_balance = max(self.peak_balance, self.balance)

        elif action == 0:                               # 홀드
            if self.position == 0:
                reward = -0.0001    # 무포지션 관망 패널티
            else:
                # ④ 24h 이후 가중 보유 패널티 (선형 증가)
                #    - 24h(288스텝) 이내: base_penalty 고정
                #    - 24h 초과 이후: 24h 추가마다 배율 +1 선형 증가
                #      (48h: ×2, 72h: ×3, 1주: ×8)
                base_penalty = -0.0005
                if hold_steps >= HOLD_BASE_STEPS:
                    extra  = (hold_steps - HOLD_BASE_STEPS) / HOLD_BASE_STEPS
                    reward = base_penalty * (1.0 + extra)
                else:
                    reward = base_penalty

        # ⑤ 드로우다운 패널티: peak 대비 -10% 초과 시 초과분에 비례
        #    계수 0.01: 20% DD → 0.001/step, 20,000스텝 × 0.001 = 20 (매우 약한 패널티)
        drawdown = (self.peak_balance - self.balance) / (self.peak_balance + 1e-8)
        if drawdown > DD_THRESHOLD:
            reward -= (drawdown - DD_THRESHOLD) * 0.001

        # ⑥ per-step 보상 클리핑: 이상치 방지 (critic 안정화)
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

        self.current_step += 1
        ep_steps = self.current_step - self.start_step

        if self.balance <= 0:
            terminated = True
        if self.current_step >= self.max_steps:
            truncated = True
        elif self.max_episode_steps is not None and ep_steps >= self.max_episode_steps:
            truncated = True

        if terminated or truncated:
            # 에피소드 종료 시 미청산 포지션 강제 청산 + 패널티
            if self.position != 0:
                final_price = self.closes[min(self.current_step, self.max_steps - 1)]
                if self.position == 1:
                    forced_ret = (final_price - self.entry_price) / self.entry_price
                else:
                    forced_ret = (self.entry_price - final_price) / self.entry_price
                net_ret = forced_ret - (self.fee_rate * 2)
                self.balance = self.balance * (1 + forced_ret) * (1 - self.fee_rate)
                self.total_trades += 1
                if net_ret > 0:
                    self.win_trades += 1
                reward += -2.0  # 강제 청산 벌점
                self.position = 0

            info = {
                'final_balance': self.balance,
                'total_trades':  self.total_trades,
                'win_rate': (self.win_trades / self.total_trades * 100)
                            if self.total_trades > 0 else 0.0,
            }

        return self._get_obs(), reward, terminated, truncated, info