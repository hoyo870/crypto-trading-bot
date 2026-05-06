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
MAX_EP_STEPS    = 20_000 # 에피소드 최대 길이 (약 70일)
REWARD_CLIP     = 10.0   # per-step 보상 클리핑 범위
IMBALANCE_FREE_BAND = 0.60
IMBALANCE_PENALTY_COEF = 0.30
TAIL_LOSS_THRESHOLD = 0.01  # 순손실 1% 초과분부터 꼬리손실 패널티
TAIL_LOSS_COEF = 0.20

# ── 레버리지 설정 ────────────────────────────────────────────────
# DEFAULT_LEVERAGE: 환경 생성 시 기본값.
#   1 → 현물(v2와 동일), 2 → 2배, 3 → 3배
# 청산 조건: 포지션 수익률 <= -1/leverage (마진 전액 소진)
# 펀딩비  : 포지션 유지 비용 (8시간마다 0.01% → 5분봉 1스텝당 ~0.00000417%)
DEFAULT_LEVERAGE = 2
FUNDING_RATE_PER_STEP = 0.0001 / (8 * 12)  # 0.01% / (8h × 12 steps/h)
# ────────────────────────────────────────────────────────────────


class LeverageTradingEnv(gym.Env):
    """
    레버리지 선물 거래 환경 (v3)

    v2 CryptoTradingEnv 대비 추가/변경 사항:
    - leverage 파라미터: 포지션 수익률에 배수 적용
    - 청산(liquidation): 수익률 <= -1/leverage 이면 즉시 청산 + 강패널티
    - 펀딩비: 포지션 유지 중 매 스텝 FUNDING_RATE_PER_STEP × leverage 차감
    - obs: 기존 9차원 + leverage_norm(1차원) = 10차원
      leverage_norm = (leverage - 1) / 4  → [1배=0, 5배=1] 범위 정규화
    - 보상: 레버리지 배수에 비례한 수익이지만 MDD 패널티 강화 (계수 ×leverage)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, initial_balance=10000.0,
                 fee_rate=0.0005, leverage=DEFAULT_LEVERAGE, mode=None):
        super().__init__()
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.leverage = float(leverage)

        # 레버리지에 따른 청산 임계치: -1/leverage
        self.liq_threshold = -1.0 / self.leverage

        print(f"[INFO] LeverageTradingEnv 데이터 로드 중 (레버리지={int(self.leverage)}x)...")
        df = pd.read_csv(data_path)
        self.max_steps = len(df) - 1

        self.closes         = df['close'].values.astype(np.float32)
        self.long_scores    = df['long_score'].values.astype(np.float32)
        self.short_scores   = df['short_score'].values.astype(np.float32)
        self.context_scores = df['context_score'].values.astype(np.float32)

        dt = pd.to_datetime(df['datetime'])
        hours = dt.dt.hour.values
        dows  = dt.dt.dayofweek.values
        self.hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
        self.hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
        self.dow_sin  = np.sin(2 * np.pi * dows  /  7).astype(np.float32)
        self.dow_cos  = np.cos(2 * np.pi * dows  /  7).astype(np.float32)

        self.action_space = spaces.Discrete(4)
        # obs(10): long_score, short_score, context_score, position, unrealized_pnl,
        #          hour_sin, hour_cos, dow_sin, dow_cos, leverage_norm
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

    # ────────────────────── reset ──────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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

        self.balance       = self.initial_balance
        self.peak_balance  = self.initial_balance
        self.position      = 0
        self.entry_price   = 0.0
        self.entry_step    = 0
        self.total_trades  = 0
        self.win_trades    = 0
        self.long_entries  = 0
        self.short_entries = 0
        self.recent_returns = []
        self.liquidated    = False
        return self._get_obs(), {}

    # ────────────────────── obs ────────────────────────────────
    def _get_obs(self):
        i = self.current_step
        current_price = self.closes[i]
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.leverage
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price * self.leverage

        leverage_norm = float((self.leverage - 1.0) / 4.0)  # 1배=0, 5배=1

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
            leverage_norm,
        ], dtype=np.float32)

    def _apply_funding(self):
        """포지션 유지 중 매 스텝 펀딩비 차감 (선물 포지션 유지 비용)"""
        if self.position != 0:
            funding = self.balance * FUNDING_RATE_PER_STEP * self.leverage
            self.balance -= funding

    def _check_liquidation(self, current_price):
        """
        레버리지 청산 조건 확인.
        수익률이 -1/leverage 이하면 True 반환.
        """
        if self.position == 0:
            return False
        if self.position == 1:
            raw_ret = (current_price - self.entry_price) / self.entry_price
        else:
            raw_ret = (self.entry_price - current_price) / self.entry_price
        return raw_ret <= self.liq_threshold

    # ────────────────────── step ───────────────────────────────
    def step(self, action):
        i = self.current_step
        current_price = self.closes[i]
        reward     = 0.0
        terminated = False
        truncated  = False
        info       = {}

        hold_steps = (self.current_step - self.entry_step) if self.position != 0 else 0

        # ① 강제청산: 레버리지 마진 소진 감지 (청산 우선 처리)
        if self._check_liquidation(current_price):
            # 청산 시 잔액 = 0 처리 (마진 전액 손실)
            self.total_trades += 1
            liq_reward = -REWARD_CLIP  # 최대 패널티
            reward = liq_reward
            self.balance = 0.0
            self.position = 0
            self.liquidated = True
            terminated = True
            info = {
                'final_balance': self.balance,
                'total_trades': self.total_trades,
                'win_rate': (self.win_trades / self.total_trades * 100)
                            if self.total_trades > 0 else 0.0,
                'liquidated': True,
            }
            return self._get_obs(), reward, terminated, truncated, info

        # ② 1주 초과 보유 강제 청산
        if self.position != 0 and hold_steps >= MAX_HOLD_STEPS:
            action = 3

        # 최소 보유 전 청산 무효
        if action == 3 and self.position != 0 and hold_steps < MIN_HOLD_STEPS:
            action = 0
            reward -= 0.02

        # ── 행동 처리 ──
        if action == 1 and self.position == 0:          # Long 진입
            self.position    = 1
            self.entry_price = current_price
            self.entry_step  = self.current_step
            # 진입 수수료: 레버리지 포지션 명목가액 기준
            self.balance    *= (1 - self.fee_rate * self.leverage)
            self.long_entries += 1

            total_entries = self.long_entries + self.short_entries
            imbalance = abs(self.long_entries - self.short_entries) / max(1, total_entries)
            if total_entries >= 10 and imbalance > IMBALANCE_FREE_BAND:
                reward -= (imbalance - IMBALANCE_FREE_BAND) * IMBALANCE_PENALTY_COEF

        elif action == 2 and self.position == 0:        # Short 진입
            self.position    = -1
            self.entry_price = current_price
            self.entry_step  = self.current_step
            self.balance    *= (1 - self.fee_rate * self.leverage)
            self.short_entries += 1

            total_entries = self.long_entries + self.short_entries
            imbalance = abs(self.long_entries - self.short_entries) / max(1, total_entries)
            if total_entries >= 10 and imbalance > IMBALANCE_FREE_BAND:
                reward -= (imbalance - IMBALANCE_FREE_BAND) * IMBALANCE_PENALTY_COEF

        elif action == 3 and self.position != 0:        # 청산
            if self.position == 1:
                raw_ret = (current_price - self.entry_price) / self.entry_price
            else:
                raw_ret = (self.entry_price - current_price) / self.entry_price

            lev_ret = raw_ret * self.leverage
            self.balance = self.balance * (1 + lev_ret) * (1 - self.fee_rate * self.leverage)
            self.total_trades += 1
            net_ret = lev_ret - (self.fee_rate * self.leverage * 2)

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
                tail_loss = max(0.0, abs(net_ret) - TAIL_LOSS_THRESHOLD * self.leverage)
                tail_penalty = ((tail_loss * 100.0) ** 2) * TAIL_LOSS_COEF
                reward = (net_ret * 100.0) - 1.5 + sharpe_bonus - tail_penalty

            self.position = 0
            self.peak_balance = max(self.peak_balance, self.balance)

        elif action == 0:                               # 홀드
            if self.position == 0:
                reward = -0.0001
            else:
                base_penalty = -0.0005
                if hold_steps >= HOLD_BASE_STEPS:
                    extra  = (hold_steps - HOLD_BASE_STEPS) / HOLD_BASE_STEPS
                    reward = base_penalty * (1.0 + extra)
                else:
                    reward = base_penalty

        # ③ 포지션 유지 중 펀딩비 차감
        self._apply_funding()

        # ④ 드로우다운 패널티 (레버리지 배수로 강화)
        drawdown = (self.peak_balance - self.balance) / (self.peak_balance + 1e-8)
        if drawdown > DD_THRESHOLD:
            reward -= (drawdown - DD_THRESHOLD) * 0.001 * self.leverage

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
            if self.position != 0:
                final_price = self.closes[min(self.current_step, self.max_steps - 1)]
                if self.position == 1:
                    raw_ret = (final_price - self.entry_price) / self.entry_price
                else:
                    raw_ret = (self.entry_price - final_price) / self.entry_price
                lev_ret = raw_ret * self.leverage
                net_ret = lev_ret - (self.fee_rate * self.leverage * 2)
                self.balance = self.balance * (1 + lev_ret) * (1 - self.fee_rate * self.leverage)
                self.total_trades += 1
                if net_ret > 0:
                    self.win_trades += 1
                reward += -2.0
                self.position = 0

            info = {
                'final_balance': self.balance,
                'total_trades': self.total_trades,
                'win_rate': (self.win_trades / self.total_trades * 100)
                            if self.total_trades > 0 else 0.0,
                'liquidated': self.liquidated,
            }

        return self._get_obs(), reward, terminated, truncated, info
