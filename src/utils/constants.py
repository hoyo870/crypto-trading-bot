"""
src/utils/constants.py
프로젝트 전역 날짜/분할 상수.

기존 분산 정의 현황:
  - scripts/00_prepare_data.py : PHASE_ACCUM_END, PHASE_BULL_END, PHASE_VAL_END
  - src/models/base_models.py  : _PHASE_BULL_END, _PHASE_VAL_END  (동일값, 중복)

이 파일에서 단일 정의하고, 나머지 모듈에서 import 해 사용합니다.

시장 구분:
  Phase 0 (Accumulation) : 2023-05-01  ~  PHASE_ACCUM_END
  Phase 1 (Bull)         : PHASE_ACCUM_END+1step  ~  PHASE_BULL_END  ← train 기준
  Phase 2 (Bear/Unknown) : PHASE_BULL_END+1step  ~  (미래)

데이터 분할:
  train : ~ PHASE_BULL_END
  val   : PHASE_BULL_END+1step  ~  PHASE_VAL_END
  test  : PHASE_VAL_END+1step  ~  (이후 전부)
"""

import pandas as pd

# ── 시장 페이즈 경계 ──────────────────────────────────────────────────────
PHASE_ACCUM_END = pd.Timestamp('2023-10-15 23:59:00')   # Phase 0 끝
PHASE_BULL_END  = pd.Timestamp('2025-06-30 23:59:00')   # Phase 1 끝 / train 정규화 기준
PHASE_VAL_END   = pd.Timestamp('2025-10-31 23:59:00')   # val 끝

# ── Base Expert 훈련 전용 ────────────────────────────────────────────────
# Long 전문가 도메인 시프트 완화 전략:
#   - 일반 분할: Train = Phase 0+1 (Bull), Val = Phase 2 (Bear) → AUC 0.5411 (랜덤 수준)
#   - Long 전용 분할: Train = Phase 0+1 초중반(~LONG_TRAIN_END), Val = Phase 1 후반
#     → Train/Val 모두 Bull 레짐 → 도메인 일치, 신호 품질 향상 기대
# 사용: python 01_train_base.py --only long --long-split
LONG_TRAIN_END  = pd.Timestamp('2024-06-30 23:59:00')   # Long 전용 train 끝 (Bull 초중반)
LONG_VAL_END    = PHASE_BULL_END                         # Long 전용 val 끝 = Bull 끝

# ── 하위 호환 별칭 (base_models.py 에서 _ prefix 로 정의되던 이름) ──────
_PHASE_BULL_END = PHASE_BULL_END
_PHASE_VAL_END  = PHASE_VAL_END
