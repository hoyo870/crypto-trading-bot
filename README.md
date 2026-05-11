# Crypto Trading Bot

강화학습(RL) 기반 암호화폐 레버리지 선물 트레이딩 연구·실험 저장소입니다.  
Base Expert(LSTM) → 신호 추출 → PPO RL Agent → 자동 세대 진화 파이프라인으로 구성됩니다.

---

## Repository Layout

```
.
├── run_evolution.py          # ★ 원클릭 진화 파이프라인 오케스트레이터
├── scripts/
│   ├── 01_train_base.py      # Base Expert (Long/Short/Context LSTM) 학습
│   ├── 02_extract_signals.py # 신호 추출 (long_score / short_score / context_score)
│   ├── 03_train_rl.py        # RL Agent (PPO) 단일/배치 학습
│   ├── 04_train_rl_batch.py  # 병렬 훈련 큐 오케스트레이터 (M1 Max 최적화)
│   ├── 05_backtest.py        # 병렬 일괄 백테스트 및 랭킹 산출
│   └── tools/
│       └── clear_artifacts.py  # 실험 산출물 일괄 정리
├── src/
│   ├── envs/
│   │   ├── trading_env_baby.py  # ★ 커리큘럼 학습 전용 경량 환경 (현재 사용)
│   │   └── trading_env.py       # 풀 패널티 환경 (레거시)
│   └── models/
│       └── base_models.py       # PriceActionExpert / ContextExpert / prepare_expert_data
├── checkpoints/
│   ├── base_experts/            # long/short/context expert .pth
│   └── rl_generations/
│       ├── gen1/                # 1세대 RL 모델 (best_gen1.zip 포함)
│       ├── gen2/
│       └── ...
├── data/
│   ├── raw/                     # OHLCV 5분봉 원본 (BTC/ETH/SOL/XRP)
│   ├── processed/               # TA 지표 포함 가공 데이터
│   └── signals/                 # 03 Expert 추론 결과 (RL 학습 입력)
├── reports/
│   ├── gen1/                    # 세대별 백테스트 차트 및 요약 JSON
│   └── ...
├── logs/
│   ├── orchestrator.log         # 전체 통합 로그
│   └── train/
│       └── gen1/                # 세대별 분리 로그 (train.log / batch.log / backtest.log)
└── requirements.txt
```

---

## Setup

### 1) Python 환경 (conda 권장)

```bash
conda create -n cryptobot python=3.10
conda activate cryptobot
pip install -r requirements.txt
```

### 2) TA-Lib 설치 (macOS)

```bash
brew install ta-lib
pip install TA-Lib
```

---

## 실행 흐름 (01 → 02 → 진화 파이프라인)

### Step 01 — Base Expert 학습

Long / Short / Context 3개의 LSTM 신호 전문가를 학습합니다.

```bash
# BTC 기본
python scripts/01_train_base.py

# 다른 심볼 지정
python scripts/01_train_base.py --symbol ETH_USDT
python scripts/01_train_base.py --symbol SOL_USDT
```

산출물: `checkpoints/base_experts/{long,short,context}_expert.pth`

---

### Step 02 — Base Signal 추출

학습된 Expert로 전체 시계열에 대한 점수를 추론해 RL 학습용 신호 파일을 생성합니다.

```bash
# BTC 기본 (출력: data/signals/BTC_USDT_signals_log.csv)
python scripts/02_extract_signals.py

# 다른 심볼
python scripts/02_extract_signals.py --symbol ETH_USDT
# 출력: data/signals/ETH_USDT_signals_log.csv
```

산출물: `data/signals/{symbol}_signals_log.csv`

---

### Step 03 — 진화 파이프라인 (★ 메인)

`run_evolution.py`가 훈련(04) → 백테스트(05) → 자동 폐기 → 다음 세대 이식을 자동 반복합니다.

```bash
# 시나리오 1: 1세대만 가볍게 (각 조합 10개, 전부 보존)
python run_evolution.py --target-generations 1 --count-per-task 10 --auto-discard-top 999

# 시나리오 2: 3세대 연속 진화 (1등만 살아남아 다음 세대 부모로 이식)
python run_evolution.py \
  --target-generations 3 \
  --auto-discard-top 1 \
  --leverages 1,3,5 \
  --profiles stable,balanced,aggressive \
  --count-per-task 33

# 시나리오 3: 과거 모델에서 파인튜닝 시작
python run_evolution.py \
  --initial-parent checkpoints/rl_generations/gen1/best_gen1.zip \
  --target-generations 2
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--target-generations` | 1 | 진화시킬 세대 수 |
| `--leverages` | `1,3,5` | 레버리지 목록 (쉼표 구분) |
| `--profiles` | `stable,balanced,aggressive` | 하이퍼파라미터 프로파일 |
| `--count-per-task` | 10 | (레버리지 × 프로파일) 조합당 학습 모델 수 |
| `--jobs` | 3 | 병렬 프로세스 수 |
| `--auto-discard-top` | 3 | 백테스트 상위 K개만 보존, 나머지 삭제 |
| `--initial-parent` | None | 1세대 시작 시 부모 가중치 경로 (파인튜닝) |

---

### 개별 스크립트 직접 실행 (고급)

```bash
# 단일 RL 학습
python scripts/03_train_rl.py \
  --data-path data/signals/BTC_USDT_signals_log.csv \
  --leverage 2 --tuning-profile balanced \
  --count 1 --timesteps 3000000

# 병렬 배치 학습
python scripts/04_train_rl_batch.py \
  --leverages 1,3 --profiles stable,balanced \
  --count-per-task 5 --jobs 3

# 백테스트
python scripts/05_backtest.py \
  --model-dir checkpoints/rl_generations/gen1 \
  --data-path data/signals/BTC_USDT_signals_log.csv \
  --reports-dir reports/gen1

# 산출물 정리 (dry-run)
python scripts/tools/clear_artifacts.py --dry-run
python scripts/tools/clear_artifacts.py --targets logs,reports --yes
```

---

## 환경 구조 (BabyLeverageTradingEnv)

- **Action Space**: `Discrete(6)` — 0=홀드, 1=롱Full, 2=롱Half, 3=숏Full, 4=숏Half, 5=청산
- **Observation**: `Box(13,)` — long/short/context score, 포지션, 미실현 손익, 시간 피처, 레버리지, 홀드비율, DD
- **Train/Eval 분리**: `mode="train"` → 전체 데이터의 앞 70% | `mode="eval"` → 뒤 30%
- **Max Episode**: 20,000 스텝 (~70일, 5분봉 기준)
- **청산 벌점**: LIQ_PENALTY = 100.0 (커리큘럼 완화)

---

## Notes

- `requirements.txt`의 `torch` / `stable-baselines3` 버전을 변경하면 저장된 모델 직렬화 호환성이 깨질 수 있습니다.
- 세대별 로그는 `logs/train/gen{N}/` 에 `train.log` / `batch.log` / `backtest.log`로 분리 저장됩니다.
- `--auto-discard-top 999`로 설정하면 폐기 없이 모든 산출물을 보존할 수 있습니다.

