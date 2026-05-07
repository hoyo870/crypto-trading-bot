import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

# ── 하이퍼파라미터 ───────────────────────────────────────────────
MIN_HOLD_STEPS      = 12       # 최소 보유: 1시간 (12 × 5분봉)
MAX_HOLD_STEPS_BASE = 2016     # 기본 최대 보유: 1주 (실제값 = base / leverage)
HOLD_BASE_STEPS     = 288      # 보유 시간 패널티 시작: 24시간
SHARPE_WINDOW       = 50       # Sharpe 계산용 최근 거래 수
DD_THRESHOLD        = 0.10     # 드로우다운 패널티 발동: 계좌 -10%
DD_PENALTY_COEF     = 0.01     # 드로우다운 패널티 계수 (v3 0.001 × 10)
MIN_EP_STEPS        = 10_000   # 에피소드 최소 잔여 스텝
MAX_EP_STEPS        = 20_000   # 에피소드 최대 길이 (~70일)
REWARD_CLIP         = 10.0     # per-step 보상 클리핑 범위
IMBALANCE_FREE_BAND    = 0.60
IMBALANCE_PENALTY_COEF = 1.0   # 롱/숏 편향 패널티 계수 (v3 0.30 → 1.0)
TAIL_LOSS_THRESHOLD    = 0.01  # 꼬리손실 패널티 발동 순손실 기준
TAIL_LOSS_COEF         = 0.20
TRAILING_REWARD_COEF   = 0.1   # 수익 중 홀드 트레일링 보상 계수 (과대보상 방지)

# ── 스톱로스 & 손익비 ────────────────────────────────────────────

# 손익비 보상은 스톱로스 거리와 분리한다.
# 학습 신호는 "원시 가격 이동 raw_ret" 기준의 고정 위험 단위로 측정한다.
#   RR_RISK_UNIT_RAW=1% 이면,
#   1:2 달성 = +2% raw, 1:3 달성 = +3% raw
RR_RISK_UNIT_RAW = 0.01         # RR 계산의 기준 위험 단위: 1% raw move
RR_TARGET        = 2.0          # 1:2 손익비 목표
RR_TIER2         = 3.0          # 1:3 손익비 추가 보너스 발동 기준
RR_BONUS_COEF    = 1.5          # RR 달성 보너스 계수

# ── 미실현 손실 per-step 패널티 ──────────────────────────────────
UNREALIZED_LOSS_THRESHOLD = 0.05  # 레버리지 적용 손실 5% 초과부터
UNREALIZED_LOSS_COEF      = 0.05  # 패널티 계수

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
    레버리지 선물 거래 환경 (v4)

    v3 대비 변경 사항:
    ① 하드 스톱로스        : raw_ret <= -hard_sl_pct 에서 강제 청산 (기본 2% raw)
    ② MAX_HOLD 레버리지 비례: max(288, 2016 / leverage)
    ③ 미실현손실 per-step  : 레버리지 손실 5% 초과 시 매 스텝 패널티
    ④ DD 패널티 10배 강화  : 계수 0.001 → 0.01
    ⑤ 불균형 패널티 강화   : 0.30 → 1.0
    ⑥ obs 10→13 차원       : hold_ratio, drawdown_norm, position_size_norm 추가
    ⑦ 포지션 사이징        : Discrete(6) 액션스페이스
                              0=hold, 1=long_full, 2=long_half,
                              3=short_full, 4=short_half, 5=close
    ⑧ 손익비 보상 (1:2/1:3): RR≥2.0 달성 시 보너스, RR≥3.0 시 추가 보너스
    ⑨ 잔고 분리 회계       : margin_used 분리 추적 → 부분 포지션 P&L 정확 계산
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, initial_balance=10000.0,
                 fee_rate=0.0005, leverage=DEFAULT_LEVERAGE,
                 hard_sl_pct=0.02, mode=None):
        super().__init__()
        self.initial_balance = initial_balance
        self.fee_rate        = fee_rate
        self.leverage        = float(leverage)

        # 청산 임계치: raw_ret <= -1/leverage (마진 100% 소진)
        self.liq_threshold = -1.0 / self.leverage
        self.hard_sl_pct   = float(hard_sl_pct)   # 절대 강제 손절 기준 (raw, 기본 2%)
        # TP 기준 raw_ret: 고정 위험 단위 × RR_TARGET (1:2)
        self.rr_tp_raw     = RR_RISK_UNIT_RAW * RR_TARGET

        # 레버리지 비례 최대 보유 스텝 (1x:2016, 3x:672, 5x:403)
        self.max_hold_steps = max(288, int(MAX_HOLD_STEPS_BASE / self.leverage))

        print(f"[INFO] LeverageTradingEnv v4  lev={int(self.leverage)}x  "
              f"HardSL={self.hard_sl_pct*100:.1f}%raw  "
              f"RR_unit={RR_RISK_UNIT_RAW*100:.1f}%raw  "
              f"TP(1:2)={self.rr_tp_raw*100:.1f}%raw  "
              f"max_hold={self.max_hold_steps}bars")

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

        # 액션: 0=hold, 1=long_full, 2=long_half, 3=short_full, 4=short_half, 5=close
        self.action_space = spaces.Discrete(6)
        # obs(13): long_score, short_score, context_score, position, unrealized_pnl,
        #          hour_sin, hour_cos, dow_sin, dow_cos, leverage_norm,
        #          hold_ratio, drawdown_norm, position_size_norm
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
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

        self.balance        = self.initial_balance
        self.peak_balance   = self.initial_balance
        self.position       = 0      # 0: 없음, 1: 롱, -1: 숏
        self.position_size  = 0.0    # 진입 마진 비율 (0.0 / 0.5 / 1.0)
        self.margin_used    = 0.0    # 진입 시 사용된 마진 절대값
        self.entry_price    = 0.0
        self.entry_step     = 0
        self.total_trades   = 0
        self.win_trades     = 0
        self.long_entries   = 0
        self.short_entries  = 0
        self.recent_returns = []
        self.liquidated     = False
        return self._get_obs(), {}

    # ────────────────────── 내부 유틸 ──────────────────────────
    def _calc_raw_ret(self, current_price):
        """현재 포지션의 raw(비레버리지) 수익률 계산"""
        if self.position == 0:
            return 0.0
        if self.position == 1:
            return float((current_price - self.entry_price) / self.entry_price)
        return float((self.entry_price - current_price) / self.entry_price)

    def _close_position(self, raw_ret):
        """
        포지션 청산. margin_used 기반으로 P&L 계산.
        net_ret(레버리지 적용 후 수수료 양방 차감) 반환.
        """
        lev_ret = raw_ret * self.leverage
        pnl     = self.margin_used * lev_ret
        fee_out = self.margin_used * self.fee_rate * self.leverage
        self.balance  += pnl - fee_out
        self.balance   = max(0.0, self.balance)
        self.total_trades += 1
        net_ret = lev_ret - (self.fee_rate * self.leverage * 2)
        if net_ret > 0:
            self.win_trades += 1
        self.position      = 0
        self.position_size = 0.0
        self.margin_used   = 0.0
        self.peak_balance  = max(self.peak_balance, self.balance)
        return net_ret

    def _apply_funding(self):
        """포지션 보유 중 매 스텝 펀딩비 차감"""
        if self.position != 0 and self.margin_used > 0:
            funding = self.margin_used * FUNDING_RATE_PER_STEP * self.leverage
            self.balance = max(0.0, self.balance - funding)

    # ────────────────────── obs ────────────────────────────────
    def _get_obs(self):
        i = min(self.current_step, self.max_steps - 1)
        current_price = self.closes[i]

        # 미실현 손익 (레버리지 적용, -1~1 클리핑)
        unrealized_pnl = 0.0
        if self.position != 0:
            raw = (current_price - self.entry_price) / self.entry_price \
                  if self.position == 1 else \
                  (self.entry_price - current_price) / self.entry_price
            unrealized_pnl = float(np.clip(raw * self.leverage, -1.0, 1.0))

        # 보유 비율 (0~1)
        hold_steps = (self.current_step - self.entry_step) if self.position != 0 else 0
        hold_ratio = float(np.clip(hold_steps / max(1, self.max_hold_steps), 0.0, 1.0))

        # 계좌 드로우다운 (0~1)
        drawdown_norm = float(np.clip(
            (self.peak_balance - self.balance) / (self.peak_balance + 1e-8),
            0.0, 1.0
        ))

        leverage_norm      = float((self.leverage - 1.0) / 4.0)  # 1x=0, 5x=1
        position_size_norm = float(self.position_size)             # 0, 0.5, 1.0

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

    # ────────────────────── step ───────────────────────────────
    def step(self, action):
        i             = min(self.current_step, self.max_steps - 1)
        current_price = self.closes[i]
        reward        = 0.0
        terminated    = False
        truncated     = False
        info          = {}

        hold_steps = (self.current_step - self.entry_step) if self.position != 0 else 0

        # ── ① 레버리지 청산 & 하드 스톱로스 우선 처리 ─────────────
        if self.position != 0:
            raw_ret_now = self._calc_raw_ret(current_price)

            # 마진 100% 소진 → 강제 청산
            if raw_ret_now <= self.liq_threshold:
                self.balance = max(0.0, self.balance - self.margin_used)
                self.total_trades += 1
                self.position      = 0
                self.position_size = 0.0
                self.margin_used   = 0.0
                self.liquidated    = True
                terminated         = True
                reward             = -REWARD_CLIP
                info = {
                    'final_balance': self.balance,
                    'total_trades':  self.total_trades,
                    'win_rate': (self.win_trades / self.total_trades * 100)
                                if self.total_trades > 0 else 0.0,
                    'liquidated': True,
                }
                return self._get_obs(), float(reward), terminated, truncated, info

            # 절대 강제 손절: raw 손실 hard_sl_pct 초과 → reward -5.0
            if raw_ret_now <= -self.hard_sl_pct:
                net_ret = self._close_position(raw_ret_now)
                reward  = min(-5.0, (net_ret * 100.0) - 3.0)
                self.current_step += 1
                ep_steps = self.current_step - self.start_step
                if self.balance <= 0:
                    terminated = True
                elif self.current_step >= self.max_steps:
                    truncated = True
                elif self.max_episode_steps is not None and ep_steps >= self.max_episode_steps:
                    truncated = True
                info = {
                    'hard_stop_loss': True,
                    'final_balance': self.balance,
                    'total_trades':  self.total_trades,
                    'win_rate': (self.win_trades / self.total_trades * 100)
                                if self.total_trades > 0 else 0.0,
                    'liquidated': self.liquidated,
                }
                return self._get_obs(), float(reward), terminated, truncated, info

        # ── ② 최대 보유 도달 → 강제 청산 액션 ─────────────────────
        if self.position != 0 and hold_steps >= self.max_hold_steps:
            action = 5

        # 최소 보유 전 청산 무효
        if action == 5 and self.position != 0 and hold_steps < MIN_HOLD_STEPS:
            action = 0
            reward -= 0.02

        # ── ③ 액션 처리 ─────────────────────────────────────────
        if action in (1, 2) and self.position == 0:         # Long 진입 (full/half)
            margin_ratio       = MARGIN_FULL if action == 1 else MARGIN_HALF
            self.margin_used   = self.balance * margin_ratio
            self.position      = 1
            self.position_size = margin_ratio
            self.entry_price   = current_price
            self.entry_step    = self.current_step
            fee_in = self.margin_used * self.fee_rate * self.leverage
            self.balance = max(0.0, self.balance - fee_in)
            self.long_entries += 1
            total_entries = self.long_entries + self.short_entries
            imbalance = abs(self.long_entries - self.short_entries) / max(1, total_entries)
            if total_entries >= 10 and imbalance > IMBALANCE_FREE_BAND:
                reward -= (imbalance - IMBALANCE_FREE_BAND) * IMBALANCE_PENALTY_COEF

        elif action in (3, 4) and self.position == 0:       # Short 진입 (full/half)
            margin_ratio       = MARGIN_FULL if action == 3 else MARGIN_HALF
            self.margin_used   = self.balance * margin_ratio
            self.position      = -1
            self.position_size = margin_ratio
            self.entry_price   = current_price
            self.entry_step    = self.current_step
            fee_in = self.margin_used * self.fee_rate * self.leverage
            self.balance = max(0.0, self.balance - fee_in)
            self.short_entries += 1
            total_entries = self.long_entries + self.short_entries
            imbalance = abs(self.long_entries - self.short_entries) / max(1, total_entries)
            if total_entries >= 10 and imbalance > IMBALANCE_FREE_BAND:
                reward -= (imbalance - IMBALANCE_FREE_BAND) * IMBALANCE_PENALTY_COEF

        elif action == 5 and self.position != 0:             # 자발적 청산
            raw_ret = self._calc_raw_ret(current_price)
            net_ret = self._close_position(raw_ret)

            # Sharpe 보너스
            self.recent_returns.append(net_ret)
            if len(self.recent_returns) > SHARPE_WINDOW:
                self.recent_returns.pop(0)
            sharpe_bonus = 0.0
            if len(self.recent_returns) >= 2:
                sharpe_bonus = (net_ret / (np.std(self.recent_returns) + 1e-8)) * 0.5

            if net_ret > 0:
                base_reward = (net_ret * 100.0) + 1.0
                # ── 손익비 보상 (1:2 / 1:3 계단식) ──────────────
                rr_achieved = abs(raw_ret) / (RR_RISK_UNIT_RAW + 1e-8)
                if rr_achieved >= RR_TIER2:      # 1:3 달성
                    rr_bonus = (rr_achieved - RR_TIER2 + 2.0) * RR_BONUS_COEF
                elif rr_achieved >= RR_TARGET:   # 1:2 달성
                    rr_bonus = (rr_achieved - RR_TARGET + 1.0) * RR_BONUS_COEF
                else:
                    rr_bonus = 0.0
                reward = base_reward + sharpe_bonus + rr_bonus
            else:
                tail_loss    = max(0.0, abs(net_ret) - TAIL_LOSS_THRESHOLD * self.leverage)
                tail_penalty = ((tail_loss * 100.0) ** 2) * TAIL_LOSS_COEF
                reward = (net_ret * 100.0) - 1.5 + sharpe_bonus - tail_penalty

        elif action == 0:                                    # 홀드
            if self.position == 0:
                # 규칙3: 무포지션 관망 페널티 2배 상향 (-0.0001 → -0.0002)
                reward = -0.0002
            else:
                raw_ret = self._calc_raw_ret(current_price)
                lev_ret = raw_ret * self.leverage
                if lev_ret > 0:
                    # 규칙2: 수익 중 → 시간 페널티 면제 + 트레일링 보상
                    reward = lev_ret * TRAILING_REWARD_COEF
                else:
                    # 손실 중 또는 손익분기: 기존 패널티 유지
                    unreal_penalty = 0.0
                    if lev_ret < -UNREALIZED_LOSS_THRESHOLD:
                        unreal_penalty = abs(lev_ret) * UNREALIZED_LOSS_COEF * self.position_size
                    base_penalty = -0.0005 * max(1.0, self.leverage * self.position_size)
                    if hold_steps >= HOLD_BASE_STEPS:
                        extra  = (hold_steps - HOLD_BASE_STEPS) / HOLD_BASE_STEPS
                        reward = base_penalty * (1.0 + extra ** 1.5) - unreal_penalty
                    else:
                        reward = base_penalty - unreal_penalty

        # ── ④ 펀딩비 차감 ──────────────────────────────────────
        self._apply_funding()

        # ── ⑤ 드로우다운 패널티 (10배 강화, 레버리지 비례) ────────
        drawdown = (self.peak_balance - self.balance) / (self.peak_balance + 1e-8)
        if drawdown > DD_THRESHOLD:
            reward -= (drawdown - DD_THRESHOLD) * DD_PENALTY_COEF * self.leverage

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
