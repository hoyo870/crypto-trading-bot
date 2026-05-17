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
BREAKEVEN_TRIGGER_PCT  = 0.01  # 미실현 수익 1% 달성 후 수익이 0 이하로 회귀하면 강제 청산 (본절컷)

# ── 포지션 사이징 ─────────────────────────────────────────────────
# 액션: 0=hold, 1=long_full, 2=long_half, 3=short_full, 4=short_half, 5=close
MARGIN_FULL = 1.0   # 잔고 100%를 마진으로 사용
MARGIN_HALF = 0.5   # 잔고 50%를 마진으로 사용 (레버리지 > 1 시 위험 분산)

# ── 레버리지 설정 ─────────────────────────────────────────────────
DEFAULT_LEVERAGE      = 2
FUNDING_RATE_PER_STEP = 0.0001 / (8 * 12)   # 0.01% / 8h, 5분봉 기준
# ─────────────────────────────────────────────────────────────────

TUNING_PROFILES = {
    "stable": {
        "hold_profit_peak_bonus": 0.0000,
        "hold_loss_base_penalty": -0.00025,
        "loss_time_exp_rate": 2.0,
        "drawdown_threshold": 0.04,
        "drawdown_penalty_coef": 0.12,
    },
    "balanced": {
        "hold_profit_peak_bonus": 0.0002,
        "hold_loss_base_penalty": -0.00030,
        "loss_time_exp_rate": 2.8,
        "drawdown_threshold": 0.05,
        "drawdown_penalty_coef": 0.10,
    },
    "aggressive": {
        "hold_profit_peak_bonus": 0.0005,
        "hold_loss_base_penalty": -0.00035,
        "loss_time_exp_rate": 3.4,
        "drawdown_threshold": 0.07,
        "drawdown_penalty_coef": 0.08,
    },
}

class LeverageTradingEnv(gym.Env):
    """
    레버리지 선물 거래 환경 (v6 튜닝 프로파일 지원)

    보상 설계 핵심:
    1) 강제 청산은 즉시 종료하며 큰 음수 보상을 부여
    2) 홀드 보상은 손익 상태와 보유 시간에 따라 차등 부여
    3) 손실 홀딩 시 손실 깊이/보유시간을 반영한 패널티 적용
    4) 계좌 드로우다운 임계치 초과 시 추가 패널티 적용
    5) 롱/숏 편향이 심해지면 진입 시 추가 패널티 적용
    6) 튜닝 프로파일: stable / balanced / aggressive
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path=None, initial_balance=10000.0,
                 fee_rate=0.0005, leverage=DEFAULT_LEVERAGE,
                 mode=None, tuning_profile="balanced",
                 df=None, train_ratio=0.7, eval_ratio=0.3):
        super().__init__()
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.leverage = float(leverage)

        # 청산 임계치: raw_ret <= -1/leverage (마진 100% 소진)
        self.liq_threshold = -1.0 / self.leverage

        # 최대 보유 스텝: 24시간 고정 (레버리지 무관)
        self.max_hold_steps = MAX_HOLD_STEPS_BASE

        if tuning_profile not in TUNING_PROFILES:
            raise ValueError(f"Unknown tuning_profile: {tuning_profile}")
        self.tuning_profile = tuning_profile
        p = TUNING_PROFILES[tuning_profile]
        self.hold_profit_peak_bonus = float(p["hold_profit_peak_bonus"])
        self.hold_loss_base_penalty = float(p["hold_loss_base_penalty"])
        self.loss_time_exp_rate = float(p["loss_time_exp_rate"])
        self.drawdown_threshold = float(p["drawdown_threshold"])
        self.drawdown_penalty_coef = float(p["drawdown_penalty_coef"])

        print(f"[INFO] LeverageTradingEnv v6 lev={int(self.leverage)}x  "
              f"liq_at={1/self.leverage*100:.0f}%raw  "
              f"max_hold={self.max_hold_steps}bars  "
              f"SafeLoss={SAFE_LOSS_THRESHOLD*100:.0f}%  "
              f"profile={self.tuning_profile}")

        # ── 데이터 로드 (df 우선, 없으면 data_path 에서 CSV 로드) ──────────
        if df is None:
            if data_path is None:
                raise ValueError(
                    "[LeverageTradingEnv] data_path 또는 df 중 하나는 반드시 제공해야 합니다."
                )
            df = pd.read_csv(data_path)
        else:
            df = df.reset_index(drop=True)

        # mode 기반 데이터 분할 (df 전달 시에만 적용 — data_path 는 전구간 사용)
        if df is not None and mode in ("train", "eval") and data_path is None:
            n = len(df)
            if mode == "train":
                df = df.iloc[:int(n * train_ratio)].reset_index(drop=True)
            else:  # eval
                df = df.iloc[int(n * train_ratio):].reset_index(drop=True)

        self.max_steps = len(df) - 1

        self.closes = df['close'].values.astype(np.float32)
        self.long_scores = df['long_score'].values.astype(np.float32)
        self.short_scores = df['short_score'].values.astype(np.float32)
        self.context_scores = df['context_score'].values.astype(np.float32)

        dt = pd.to_datetime(df['datetime'])
        hours = dt.dt.hour.values
        dows = dt.dt.dayofweek.values
        self.hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
        self.hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
        self.dow_sin = np.sin(2 * np.pi * dows / 7).astype(np.float32)
        self.dow_cos = np.cos(2 * np.pi * dows / 7).astype(np.float32)

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

    def action_masks(self) -> np.ndarray:
        """ActionMasker 용: 현재 포지션에 따라 유효한 액션 마스크.

        포지션 없음(0) → 진입(1~4)만 허용, hold(0)는 항상 유효
        포지션 보유(±1) → 청산(5)와 hold(0)만 허용
        """
        mask = np.zeros(6, dtype=bool)
        mask[0] = True  # hold 언제나 유효
        if self.position == 0:
            mask[1] = mask[2] = mask[3] = mask[4] = True  # 진입 액션
        else:
            mask[5] = True  # 청산만
        return mask

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

                reward                          = -LIQ_PENALTY

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

        # ── 본절컷(Breakeven Stop) ─────────────────────────────────
        # 미실현 수익이 한번이라도 1% 이상 올랐다가 0 이하로 내려오면 강제 청산
        _breakeven_triggered = False
        if (
            self.position != 0
            and self.peak_unrealized_profit >= BREAKEVEN_TRIGGER_PCT
            and action != 5  # 이미 청산 액션이 아닌 경우에만 강제
        ):
            current_raw_ret = self._calc_raw_ret(current_price)
            if current_raw_ret <= 0.0:
                action = 5
                _breakeven_triggered = True  # MIN_HOLD_STEPS 우회 플래그

        # 최소 보유 전 청산 무효 (단, 본절컷 발동 시에는 MIN_HOLD 우회)
        if action == 5 and self.position != 0 and hold_steps < MIN_HOLD_STEPS and not _breakeven_triggered:
            action = 0
            reward -= 0.02

        # 유효하지 않은 액션 치환 및 패널티
        is_invalid_action = False
        if action in (1, 2, 3, 4) and self.position != 0:
            is_invalid_action = True
            action = 0
        elif action == 5 and self.position == 0:
            is_invalid_action = True
            action = 0

        if is_invalid_action:
            reward -= 0.05

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
                reward += net_ret * 100.0
                # ── 승률 가중 보너스 ─────────────────────────────────
                # 이 에피소드에서 승률 > 50% 이면 보상 × 1.2
                ep_win_rate = (self.win_trades / self.total_trades
                               if self.total_trades > 0 else 0.0)
                if ep_win_rate > 0.5:
                    reward *= 1.2
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
                        reward += self.hold_profit_peak_bonus
                    else:
                        reward += -(self.peak_unrealized_profit - raw_ret) * 100.0 * 0.1
                else:
                    # 튜닝 프로파일 기반: 시간 가중치 + 손실 깊이 반영
                    time_weight = math.exp(self.loss_time_exp_rate * (hold_steps / 288.0))
                    loss_magnitude = abs(lev_ret) * 100.0
                    reward += self.hold_loss_base_penalty * time_weight * (1.0 + loss_magnitude)

        # ── ④ 펀딩비 차감 ──────────────────────────────────────
        self._apply_funding()

        # 튜닝 프로파일 기반: 드로우다운 방어 패널티
        current_drawdown = (self.peak_balance - self.balance) / (self.peak_balance + 1e-8)
        if current_drawdown > self.drawdown_threshold:
            reward -= current_drawdown * self.drawdown_penalty_coef

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