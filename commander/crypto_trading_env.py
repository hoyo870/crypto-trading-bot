import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import warnings
import math
warnings.filterwarnings('ignore')

# ── 하이퍼파라미터 ───────────────────────────────────────────────
MIN_HOLD_STEPS      = 12       # 최소 보유: 1시간 (12 × 5분봉)
MAX_HOLD_STEPS_BASE = 288      # 최대 보유: 24시간 고정 (288 × 5분봉)
MIN_EP_STEPS        = 10_000   # 에피소드 최소 잔여 스텝
MAX_EP_STEPS        = 20_000   # 에피소드 최대 길이 (~70일)
REWARD_CLIP         = 10.0     # per-step 보상 클리핑 범위 (청산 벌점 제외)
LIQ_PENALTY         = 10000.0  # 강제 청산 기준 벌점 크기(학습 보상에는 클리핑 적용)
LONG_IMBALANCE_THRESHOLD  = 0.55  # 롱 비율 임계치 (초과 시 롱 진입 패널티)
SHORT_IMBALANCE_THRESHOLD = 0.45  # 숏 비율 임계치 (초과 시 숏 진입 패널티)
IMBALANCE_PENALTY_COEF    = 1.0   # 롱/숏 편향 패널티 계수
SAFE_LOSS_THRESHOLD    = 0.02  # 손절 안전/지옥 구간 경계 (net_ret 절댓값 기준)

# ── 포지션 사이징 ─────────────────────────────────────────────────
# 액션: 0=hold, 1=long_full, 2=long_half, 3=short_full, 4=short_half, 5=close
MARGIN_FULL = 1.0   # 잔고 100%를 마진으로 사용
MARGIN_HALF = 0.5   # 잔고 50%를 마진으로 사용 (레버리지 > 1 시 위험 분산)

# ── 레버리지 설정 ─────────────────────────────────────────────────
DEFAULT_LEVERAGE      = 2
FUNDING_RATE_PER_STEP = 0.0001 / (8 * 12)   # 0.01% / 8h, 5분봉 기준
# ─────────────────────────────────────────────────────────────────

class LeverageTradingEnv(gym.Env):
    """
    레버리지 선물 거래 환경 (v4 → v5 보상 리팩토링)

    보상 설계 핵심:
    1) 강제 청산은 즉시 종료하며 큰 음수 보상을 부여
    2) 홀드 보상은 손익 상태와 보유 시간에 따라 차등 부여
    3) 손실 청산은 안전구간/초과구간으로 나눠 패널티 강도를 조절
    4) 롱/숏 편향이 심해지면 진입 시 추가 패널티 적용
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, initial_balance=10000.0,
                 fee_rate=0.0005, leverage=DEFAULT_LEVERAGE,
                 mode=None):
        super().__init__()
        self.initial_balance = initial_balance
        self.fee_rate        = fee_rate
        self.leverage        = float(leverage)

        # 청산 임계치: raw_ret <= -1/leverage (마진 100% 소진)
        self.liq_threshold = -1.0 / self.leverage

        # 최대 보유 스텝: 24시간 고정 (레버리지 무관)
        self.max_hold_steps = MAX_HOLD_STEPS_BASE

        print(f"[INFO] LeverageTradingEnv v5  lev={int(self.leverage)}x  "
              f"liq_at={1/self.leverage*100:.0f}%raw  "
              f"max_hold={self.max_hold_steps}bars  "
              f"SafeLoss={SAFE_LOSS_THRESHOLD*100:.0f}%")

        df = pd.read_csv(data_path)
        self.max_steps = len(df) - 1

        self.closes         = df['close'].values.astype(np.float32)
        self.long_scores    = df['long_score'].values.astype(np.float32)
        self.short_scores   = df['short_score'].values.astype(np.float32)
        self.context_scores = df['context_score'].values.astype(np.float32)

        dt    = pd.to_datetime(df['datetime'])
        hours = dt.dt.hour.values
        dows  = dt.dt.dayofweek.values
        self.hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
        self.hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
        self.dow_sin  = np.sin(2 * np.pi * dows  /  7).astype(np.float32)
        self.dow_cos  = np.cos(2 * np.pi * dows  /  7).astype(np.float32)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

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

        self.balance        = self.initial_balance
        self.peak_balance   = self.initial_balance
        self.position       = 0      # 0: 없음, 1: 롱, -1: 숏
        self.position_size  = 0.0    # 진입 마진 비율 (0.0 / 0.5 / 1.0)
        self.margin_used    = 0.0    # 진입 시 사용된 마진 절대값
        self.entry_price    = 0.0
        self.entry_step     = 0
        self.total_trades            = 0
        self.win_trades              = 0
        self.long_entries            = 0
        self.short_entries           = 0
        self.peak_unrealized_profit  = 0.0   
        self.liquidated              = False
        return self._get_obs(), {}

    def _calc_raw_ret(self, current_price):
        """현재 포지션 기준 비레버리지(raw) 수익률을 계산합니다."""
        if self.position == 0:
            return 0.0
        if self.position == 1:
            return float((current_price - self.entry_price) / self.entry_price)
        return float((self.entry_price - current_price) / self.entry_price)

    def _close_position(self, raw_ret):
        """포지션을 청산하고 수수료 반영 후 net_ret(레버리지 반영)을 반환합니다."""
        lev_ret = raw_ret * self.leverage
        pnl     = self.margin_used * lev_ret
        fee_out = self.margin_used * self.fee_rate * self.leverage
        self.balance  += pnl - fee_out
        self.balance   = max(0.0, self.balance)
        self.total_trades += 1
        net_ret = lev_ret - (self.fee_rate * self.leverage * 2)
        if net_ret > 0:
            self.win_trades += 1
        self.position               = 0
        self.position_size          = 0.0
        self.margin_used            = 0.0
        self.peak_unrealized_profit = 0.0
        self.peak_balance           = max(self.peak_balance, self.balance)
        return net_ret

    def _apply_funding(self):
        """포지션 보유 중 매 스텝 펀딩비를 잔고에서 차감합니다."""
        if self.position != 0 and self.margin_used > 0:
            funding = self.margin_used * FUNDING_RATE_PER_STEP * self.leverage
            self.balance = max(0.0, self.balance - funding)

    def _get_obs(self):
        i = min(self.current_step, self.max_steps - 1)
        current_price = self.closes[i]

        unrealized_pnl = 0.0
        if self.position != 0:
            raw = (current_price - self.entry_price) / self.entry_price \
                  if self.position == 1 else \
                  (self.entry_price - current_price) / self.entry_price
            unrealized_pnl = float(np.clip(raw * self.leverage, -1.0, 1.0))

        hold_steps = (self.current_step - self.entry_step) if self.position != 0 else 0
        hold_ratio = float(np.clip(hold_steps / max(1, self.max_hold_steps), 0.0, 1.0))

        drawdown_norm = float(np.clip(
            (self.peak_balance - self.balance) / (self.peak_balance + 1e-8),
            0.0, 1.0
        ))

        leverage_norm      = float((self.leverage - 1.0) / 4.0)  
        position_size_norm = float(self.position_size)             

        return np.array([
            self.long_scores[i],
            self.short_scores[i],
            self.context_scores[i],
            float(self.position),
            unrealized_pnl,
            self.hour_sin[i],
            self.hour_cos[i],
            self.dow_sin[i],
            self.dow_cos[i],
            leverage_norm,
            hold_ratio,
            drawdown_norm,
            position_size_norm,
        ], dtype=np.float32)

    def step(self, action):
        i             = min(self.current_step, self.max_steps - 1)
        current_price = self.closes[i]
        reward        = 0.0
        terminated    = False
        truncated     = False
        info          = {}

        hold_steps = (self.current_step - self.entry_step) if self.position != 0 else 0

        # ── ① 레버리지 청산 우선 처리 ─────────────────────────────
        if self.position != 0:
            raw_ret_now = self._calc_raw_ret(current_price)

            if raw_ret_now > self.peak_unrealized_profit:
                self.peak_unrealized_profit = raw_ret_now

            # 마진 100% 소진 → 강제 청산
            if raw_ret_now <= self.liq_threshold:
                self.balance = max(0.0, self.balance - self.margin_used)
                self.total_trades              += 1
                self.position                   = 0
                self.position_size              = 0.0
                self.margin_used                = 0.0
                self.peak_unrealized_profit     = 0.0
                self.liquidated                 = True
                terminated                      = True

                # 학습 안정성을 위해 즉시 클리핑 가능한 하한값으로 보상을 부여합니다.
                reward                          = -REWARD_CLIP

                info = {
                    'final_balance': self.balance,
                    'total_trades':  self.total_trades,
                    'win_rate': (self.win_trades / self.total_trades * 100)
                                if self.total_trades > 0 else 0.0,
                    'liquidated': True,
                }
                return self._get_obs(), float(reward), terminated, truncated, info

        # ── ② 최대 보유 도달 → 강제 청산 액션 ─────────────────────
        if self.position != 0 and hold_steps >= self.max_hold_steps:
            action = 5

        # 최소 보유 전 청산 무효
        if action == 5 and self.position != 0 and hold_steps < MIN_HOLD_STEPS:
            action = 0
            reward -= 0.02

        # 유효하지 않은 액션을 hold로 치환하고 패널티를 부여합니다.
        is_invalid_action = False
        if action in (1, 2, 3, 4) and self.position != 0:
            is_invalid_action = True
            action = 0  # 강제로 홀딩 상태 유지
        elif action == 5 and self.position == 0:
            is_invalid_action = True
            action = 0  # 관망 상태에서 청산 불가하므로 무시

        if is_invalid_action:
            reward -= 0.05  # 불가능한 행동을 시도한 에이전트에게 명확한 페널티 부여

        # ── ③ 액션 처리 ─────────────────────────────────────────
        if action in (1, 2) and self.position == 0:         # Long 진입
            margin_ratio                = MARGIN_FULL if action == 1 else MARGIN_HALF
            self.margin_used            = self.balance * margin_ratio
            self.position               = 1
            self.position_size          = margin_ratio
            self.entry_price            = current_price
            self.entry_step             = self.current_step
            self.peak_unrealized_profit = 0.0
            fee_in = self.margin_used * self.fee_rate * self.leverage
            self.balance = max(0.0, self.balance - fee_in)
            self.long_entries += 1
            total_entries = self.long_entries + self.short_entries
            if total_entries >= 10:
                long_ratio = self.long_entries / total_entries
                if long_ratio > LONG_IMBALANCE_THRESHOLD:
                    reward -= (long_ratio - LONG_IMBALANCE_THRESHOLD) * IMBALANCE_PENALTY_COEF

        elif action in (3, 4) and self.position == 0:       # Short 진입
            margin_ratio                = MARGIN_FULL if action == 3 else MARGIN_HALF
            self.margin_used            = self.balance * margin_ratio
            self.position               = -1
            self.position_size          = margin_ratio
            self.entry_price            = current_price
            self.entry_step             = self.current_step
            self.peak_unrealized_profit = 0.0
            fee_in = self.margin_used * self.fee_rate * self.leverage
            self.balance = max(0.0, self.balance - fee_in)
            self.short_entries += 1
            total_entries = self.long_entries + self.short_entries
            if total_entries >= 10:
                short_ratio = self.short_entries / total_entries
                if short_ratio > SHORT_IMBALANCE_THRESHOLD:
                    reward -= (short_ratio - SHORT_IMBALANCE_THRESHOLD) * IMBALANCE_PENALTY_COEF

        elif action == 5 and self.position != 0:             # 자발적 청산
            raw_ret = self._calc_raw_ret(current_price)
            net_ret = self._close_position(raw_ret)  

            if net_ret >= 0:
                reward += net_ret * 100.0  # 앞 단계 패널티를 보존하기 위해 누적 방식 사용
            else:
                abs_loss = abs(net_ret)
                if abs_loss <= SAFE_LOSS_THRESHOLD:
                    reward += net_ret * 100.0
                else:
                    excess = abs_loss - SAFE_LOSS_THRESHOLD
                    reward += net_ret * 100.0 - (excess * 100.0) ** 2 * 0.5

        elif action == 0:                                    # 홀드
            if self.position == 0:
                reward += -0.0001
            else:
                raw_ret = self._calc_raw_ret(current_price)
                lev_ret = raw_ret * self.leverage
                if lev_ret > 0:
                    if raw_ret >= self.peak_unrealized_profit:
                        reward += 0.0005 
                    else:
                        reward += -(self.peak_unrealized_profit - raw_ret) * 100.0 * 0.1
                else:
                    time_weight = math.exp(3.0 * (hold_steps / 288.0))
                    reward += -0.0003 * time_weight

        # ── ④ 펀딩비 차감 ──────────────────────────────────────
        self._apply_funding()

        # 최종 보상 클리핑
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
                raw_ret     = self._calc_raw_ret(final_price)
                self._close_position(raw_ret)
                reward += -2.0
            info = {
                'final_balance': self.balance,
                'total_trades':  self.total_trades,
                'win_rate': (self.win_trades / self.total_trades * 100)
                            if self.total_trades > 0 else 0.0,
                'liquidated': self.liquidated,
            }

        return self._get_obs(), reward, terminated, truncated, info