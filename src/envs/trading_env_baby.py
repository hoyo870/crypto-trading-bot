"""BabyLeverageTradingEnv — 커리큘럼 학습 전용 단순화 환경 (v1)

Baby 5대 원칙:
  1. action_space=Discrete(6), observation_space=Box(13,) → 기존과 완전 동일 (가중치 이전 보장)
  2. 복잡 패널티 전면 삭제: 롱/숏 비율 불균형 패널티, DD 방어 패널티,
     MIN_HOLD_STEPS, _apply_funding() 제거. MAX_HOLD_STEPS_BASE=288 강제청산만 유지.
  3. LIQ_PENALTY: 10000.0 → 100.0 (청산 공포 완화)
  4. 자발적 청산(Action 5): reward += net_ret * 100.0 선형만 (안전/지옥 구간 제거)
  5. 홀드(Action 0):
       - 무포지션: reward += -0.0001
       - 포지션 중: reward += _calc_raw_ret(price) * leverage * 0.1 (방향만 신호)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

# ── 하이퍼파라미터 ───────────────────────────────────────────────
MIN_HOLD_STEPS      = 12       # 최소 보유: 1시간 (12 × 5분봉) — 패닉 셀 방지
MAX_HOLD_STEPS_BASE = 288      # 최대 보유: 24시간 고정 (288 × 5분봉)
MIN_EP_STEPS        = 10_000   # 에피소드 최소 잔여 스텝
MAX_EP_STEPS        = 20_000   # 에피소드 최대 길이 (~70일)
REWARD_CLIP         = 10.0     # per-step 보상 클리핑 범위 (청산 벌점 제외)
LIQ_PENALTY         = 100.0    # ★ Baby: 10000 → 100 (청산 공포 완화)

# ── 포지션 사이징 ─────────────────────────────────────────────────
# 액션: 0=hold, 1=long_full, 2=long_half, 3=short_full, 4=short_half, 5=close
MARGIN_FULL = 1.0
MARGIN_HALF = 0.5

# ── 레버리지 설정 ─────────────────────────────────────────────────
DEFAULT_LEVERAGE = 2


class BabyLeverageTradingEnv(gym.Env):
    """
    레버리지 선물 거래 Baby 환경 (커리큘럼 학습 전용)

    보상 설계 (단순화):
    1) 강제 청산 → 즉시 종료, reward = -LIQ_PENALTY (100.0)
    2) 자발적 청산(Action 5) → reward += net_ret * 100.0 (순수 선형)
    3) 홀드 무포지션 → reward += -0.0001
    4) 홀드 포지션 중 → reward += raw_ret * leverage * 0.1 (미실현 방향 신호)
    5) 패널티 없음: 펀딩비, DD방어, 롱/숏 편향, MIN_HOLD_STEPS 모두 삭제
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path=None, initial_balance=10000.0,
                 fee_rate=0.0005, leverage=DEFAULT_LEVERAGE,
                 mode=None, tuning_profile="balanced",
                 df=None, train_ratio=0.7, eval_ratio=0.3):
        super().__init__()

        # ── 데이터 로드 (df 우선, 없으면 data_path 에서 CSV 로드) ─────────
        if df is None:
            if data_path is None:
                raise ValueError(
                    "[BabyEnv] data_path 또는 df 중 하나는 반드시 제공해야 합니다."
                )
            df = pd.read_csv(data_path)

        # ── 분할 비율 검증 ─────────────────────────────────────────────────
        if not (0.0 < train_ratio <= 1.0):
            raise ValueError(
                f"[BabyEnv] train_ratio 는 (0, 1] 범위여야 합니다. 입력값: {train_ratio}"
            )
        if not (0.0 < eval_ratio <= 1.0):
            raise ValueError(
                f"[BabyEnv] eval_ratio 는 (0, 1] 범위여야 합니다. 입력값: {eval_ratio}"
            )
        if train_ratio + eval_ratio > 1.0 + 1e-9:
            raise ValueError(
                f"[BabyEnv] train_ratio + eval_ratio <= 1.0 이어야 합니다. "
                f"입력값: {train_ratio} + {eval_ratio} = {train_ratio + eval_ratio:.4f}"
            )

        n = len(df)
        # ── 데이터 최소 길이 검증 ──────────────────────────────────────────
        if n < MIN_EP_STEPS * 2:
            raise ValueError(
                f"[BabyEnv] 데이터가 너무 짧습니다 (n={n}, 최소 {MIN_EP_STEPS * 2}행 필요)."
            )

        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.leverage = float(leverage)

        # 청산 임계치: raw_ret <= -1/leverage (마진 100% 소진)
        self.liq_threshold = -1.0 / self.leverage

        # 최대 보유 스텝: 24시간 고정
        self.max_hold_steps = MAX_HOLD_STEPS_BASE

        # tuning_profile 파라미터는 인터페이스 호환을 위해 수신하지만 Baby에서는 사용 안 함
        self.tuning_profile = tuning_profile
        self.mode = mode  # "train" | "eval" | None

        data_source = f"path={data_path}" if data_path else "df=<provided>"
        print(f"[INFO] BabyLeverageTradingEnv v1 lev={int(self.leverage)}x  "
              f"liq_at={1/self.leverage*100:.0f}%raw  "
              f"max_hold={self.max_hold_steps}bars  "
              f"LIQ_PENALTY={LIQ_PENALTY}  mode={mode or 'full'}  "
              f"split={train_ratio}/{eval_ratio}  {data_source}")

        self.max_steps = n - 1

        # ── Train/Eval 데이터 구간 분리 ───────────────────────────────────
        if mode == "train":
            self._step_lo = 0
            self._step_hi = int(n * train_ratio)
        elif mode == "eval":
            self._step_lo = int(n * train_ratio)
            self._step_hi = int(n * (train_ratio + eval_ratio))
            # eval 구간 최소 길이 검증
            eval_size = self._step_hi - self._step_lo
            if eval_size < MIN_EP_STEPS:
                warnings.warn(
                    f"[BabyEnv] eval 구간이 너무 짧습니다 "
                    f"({eval_size}행 < MIN_EP_STEPS={MIN_EP_STEPS}). "
                    f"전체 구간으로 fallback합니다.",
                    UserWarning,
                )
                self._step_lo = 0
                self._step_hi = n
        else:  # None: 백테스트 등 전체 구간 사용
            self._step_lo = 0
            self._step_hi = n

        self.closes         = df['close'].values.astype(np.float32)
        # Fix1: z-정규화 — raw score 범위(~0.025)를 관측 공간 [-1,1]에 고르게 분포
        # long/short: z-score /3 → [-1,1] 내 수렴 (3σ 기준)
        # context: min-max → [-1,1]
        _l_mean = float(df['long_score'].mean());  _l_std = float(df['long_score'].std()) + 1e-8
        _s_mean = float(df['short_score'].mean()); _s_std = float(df['short_score'].std()) + 1e-8
        _c_min  = float(df['context_score'].min())
        _c_rng  = float(df['context_score'].max()) - _c_min + 1e-8
        self.long_scores    = np.clip((df['long_score'].values  - _l_mean) / _l_std / 3.0, -1.0, 1.0).astype(np.float32)
        self.short_scores   = np.clip((df['short_score'].values - _s_mean) / _s_std / 3.0, -1.0, 1.0).astype(np.float32)
        self.context_scores = np.clip((df['context_score'].values - _c_min) / _c_rng * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

        dt = pd.to_datetime(df['datetime'])
        hours = dt.dt.hour.values
        dows  = dt.dt.dayofweek.values
        self.hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
        self.hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
        self.dow_sin  = np.sin(2 * np.pi * dows  /  7).astype(np.float32)
        self.dow_cos  = np.cos(2 * np.pi * dows  /  7).astype(np.float32)

        # ★ 원칙 1: action_space, observation_space 기존과 완전 동일
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

    # ── 리셋 ───────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and 'start_step' in options:
            # 외부에서 명시적으로 지정한 경우 (백테스트 등)
            self.start_step = int(options['start_step'])
        else:
            # mode 기반 구간으로 랜덤 시작점 제한 (Train/Eval 분리)
            # MAX_EP_STEPS 여백 확보: 에피소드가 _step_hi를 초과하지 않도록 보정
            available_hi = self._step_hi - MAX_EP_STEPS
            lo = max(self._step_lo, 0)
            hi = max(lo, available_hi)
            self.start_step = int(self.np_random.integers(lo, hi + 1))
        if options is not None and 'max_ep_steps' in options:
            max_ep_steps = options['max_ep_steps']
            self.max_episode_steps = None if max_ep_steps is None else int(max_ep_steps)
        else:
            self.max_episode_steps = MAX_EP_STEPS
        self.current_step = self.start_step

        self.balance       = self.initial_balance
        self.peak_balance  = self.initial_balance
        self.position      = 0       # 0: 없음, 1: 롱, -1: 숏
        self.position_size = 0.0     # 진입 마진 비율 (0.0 / 0.5 / 1.0)
        self.margin_used   = 0.0     # 진입 시 사용된 마진 절대값
        self.entry_price   = 0.0
        self.entry_step    = 0
        self.total_trades  = 0
        self.win_trades    = 0
        self.long_entries  = 0
        self.short_entries = 0
        self.liquidated    = False
        return self._get_obs(), {}

    # ── 헬퍼: 비레버리지 raw 수익률 ───────────────────────────────
    def _calc_raw_ret(self, current_price):
        if self.position == 0:
            return 0.0
        if self.position == 1:
            return float((current_price - self.entry_price) / self.entry_price)
        return float((self.entry_price - current_price) / self.entry_price)

    # ── 헬퍼: 포지션 청산 ─────────────────────────────────────────
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
        self.position      = 0
        self.position_size = 0.0
        self.margin_used   = 0.0
        self.peak_balance  = max(self.peak_balance, self.balance)
        return net_ret

    # ── 관측 벡터 (기존과 완전 동일한 13차원) ─────────────────────
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
            self.long_scores[i],       # 0
            self.short_scores[i],      # 1
            self.context_scores[i],    # 2
            float(self.position),      # 3
            unrealized_pnl,            # 4
            self.hour_sin[i],          # 5
            self.hour_cos[i],          # 6
            self.dow_sin[i],           # 7
            self.dow_cos[i],           # 8
            leverage_norm,             # 9
            hold_ratio,                # 10
            drawdown_norm,             # 11
            position_size_norm,        # 12
        ], dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        """ActionMasker 용: 현재 포지션에 따라 유효한 액션 마스크.

        포지션 없음(0) -> 진입(1~4)만 허용, hold(0)는 항상 유효
        포지션 보유(+-1) -> 청산(5)와 hold(0)만 허용
        """
        mask = np.zeros(6, dtype=bool)
        mask[0] = True  # hold 언제나 유효
        if self.position == 0:
            mask[1] = mask[2] = mask[3] = mask[4] = True  # 진입 액션
        else:
            # ★ 수정: 최소 1시간 (MIN_HOLD_STEPS) 이상 보유 시에만 청산 활성화
            hold_steps = self.current_step - self.entry_step
            if hold_steps >= MIN_HOLD_STEPS:
                mask[5] = True
        return mask

    # ── 스텝 ───────────────────────────────────────────────────────
    def step(self, action):
        i             = min(self.current_step, self.max_steps - 1)
        current_price = self.closes[i]
        reward        = 0.0
        terminated    = False
        truncated     = False
        info          = {}

        hold_steps = (self.current_step - self.entry_step) if self.position != 0 else 0

        # ── ① 레버리지 강제 청산 우선 처리 ───────────────────────
        if self.position != 0:
            raw_ret_now = self._calc_raw_ret(current_price)
            if raw_ret_now <= self.liq_threshold:
                self.balance      = max(0.0, self.balance - self.margin_used)
                self.total_trades += 1
                self.position      = 0
                self.position_size = 0.0
                self.margin_used   = 0.0
                self.liquidated    = True
                terminated         = True
                reward             = -LIQ_PENALTY  # ★ Baby: -100.0

                info = {
                    'final_balance': self.balance,
                    'total_trades':  self.total_trades,
                    'win_rate': (self.win_trades / self.total_trades * 100)
                                if self.total_trades > 0 else 0.0,
                    'liquidated': True,
                }
                return self._get_obs(), float(reward), terminated, truncated, info

        # ── ② MAX_HOLD_STEPS 도달 → 강제 close 액션으로 교체 ──────
        if self.position != 0 and hold_steps >= self.max_hold_steps:
            action = 5

        # ★ 수정: 패닉 셀(Panic Sell) 방지를 위한 최소 보유 시간 강제 (이중 안전장치)
        if action == 5 and self.position != 0 and hold_steps < MIN_HOLD_STEPS:
            action = 0
            reward -= 0.01  # 미세 무효 액션 패널티

        # ── 무효 액션 치환 ────────────────────────────────────────
        is_invalid_action = False
        if action in (1, 2, 3, 4) and self.position != 0:
            is_invalid_action = True
            action = 0
        elif action == 5 and self.position == 0:
            is_invalid_action = True
            action = 0

        if is_invalid_action:
            reward -= 0.05

        # ── ③ 액션 처리 ──────────────────────────────────────────
        if action in (1, 2) and self.position == 0:          # Long 진입
            margin_ratio       = MARGIN_FULL if action == 1 else MARGIN_HALF
            self.margin_used   = self.balance * margin_ratio
            self.position      = 1
            self.position_size = margin_ratio
            self.entry_price   = current_price
            self.entry_step    = self.current_step
            fee_in = self.margin_used * self.fee_rate * self.leverage
            self.balance = max(0.0, self.balance - fee_in)
            self.long_entries += 1
            # ★ Baby: 롱/숏 비율 불균형 패널티 삭제

        elif action in (3, 4) and self.position == 0:        # Short 진입
            margin_ratio       = MARGIN_FULL if action == 3 else MARGIN_HALF
            self.margin_used   = self.balance * margin_ratio
            self.position      = -1
            self.position_size = margin_ratio
            self.entry_price   = current_price
            self.entry_step    = self.current_step
            fee_in = self.margin_used * self.fee_rate * self.leverage
            self.balance = max(0.0, self.balance - fee_in)
            self.short_entries += 1
            # ★ Baby: 롱/숏 비율 불균형 패널티 삭제

        elif action == 5 and self.position != 0:              # 자발적 청산
            raw_ret = self._calc_raw_ret(current_price)
            net_ret = self._close_position(raw_ret)
            # ★ Baby 원칙 4: 순수 선형 보상만 (안전/지옥 구간 제거)
            reward += net_ret * 100.0

        elif action == 0:                                     # 홀드
            if self.position == 0:
                # ★ Baby 원칙 5-a: 잦은 매매 유도를 위한 관망 패널티
                reward += -0.0001
            else:
                # ★ Baby 원칙 5-b: 미실현 방향 신호 (지수감점/트레일링 삭제)
                raw_ret = self._calc_raw_ret(current_price)
                lev_ret = raw_ret * self.leverage
                reward += lev_ret * 0.1

        # ★ Baby 원칙 2: _apply_funding() 호출 삭제
        # ★ Baby 원칙 2: 드로우다운 방어 패널티 삭제

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