# Crypto Trading Bot

강화학습(RL) 기반 암호화폐 레버리지 선물 트레이딩 연구·실험 저장소입니다.  
Base Expert(LSTM) → 신호 추출 → MaskablePPO RL Agent → 자동 세대 진화 파이프라인으로 구성됩니다.

---

## Repository Layout

```
.
├── run_evolution.py              # ★ 원클릭 진화 파이프라인 오케스트레이터
├── scripts/
│   ├── 00_prepare_data.py        # Raw OHLCV → TA 지표 + 정규화 (processed CSV + scaler)
│   ├── 01_train_base.py          # Base Expert (Long/Short/Context LSTM) 학습
│   ├── 02_extract_signals.py     # 신호 추출 (long_score / short_score / context_score)
│   ├── 03_train_rl.py            # RL Agent (MaskablePPO) 단일/배치 학습
│   ├── 03b_finetune_full.py      # ★ Baby→Full 환경 커리큘럼 전이 파인튜닝
│   ├── 04_train_rl_batch.py      # 병렬 훈련 큐 오케스트레이터 (M1 Max 최적화)
│   ├── 05_backtest.py            # 단일 모델 전구간 백테스트 (MaskablePPO)
│   ├── 06_rank.py                # 백테스트 결과 랭킹 산출 + best_by_leverage.csv
│   ├── 07_backtest_batch.py      # 병렬 백테스트 배치 관리
│   └── tools/
│       ├── clear_artifacts.py    # 실험 산출물 일괄 정리
│       └── verify_base_signals.py  # Base Expert 시그널 검증 (rule-based + 4-panel chart)
├── src/
│   ├── envs/
│   │   ├── trading_env_baby.py   # ★ 커리큘럼 학습 전용 경량 환경 (Gen1~N 훈련)
│   │   └── trading_env.py        # Full 환경 (본절컷·승률 보너스·ActionMasking 포함)
│   ├── models/
│   │   └── base_models.py        # PriceActionExpert / ContextExpert / FocalLoss
│   └── utils/
│       └── platform_utils.py     # 디바이스/코어 감지 유틸
├── checkpoints/
│   ├── base_experts/             # long/short/context expert .pth
│   └── rl_generations/
│       ├── gen1/                 # 1세대 RL 모델 (best_gen1_<tag>.zip 포함)
│       ├── gen2/
│       └── ...
├── data/
│   ├── raw/                      # OHLCV 5분봉 원본 (BTC/ETH/SOL/XRP)
│   ├── processed/                # TA 지표 포함 가공 데이터
│   └── signals/                  # Expert 추론 결과 (long_score/short_score/context_score)
├── reports/
│   ├── gen1/                     # 세대별 백테스트 차트(PNG) + 요약 JSON
│   └── ...
├── logs/
│   ├── orchestrator.log          # 전체 통합 로그
│   └── train/
│       └── gen1/                 # 세대별 분리 로그 (train / batch / backtest / rank)
└── requirements.txt
```

---

## Setup

### 공통 — Python 환경 생성 (conda 권장)

```bash
conda create -n cryptobot python=3.10
conda activate cryptobot
```

---

### 🪟 Windows

#### 1) TA-Lib — 바이너리 패키지 자동 설치 (C 라이브러리 불필요)

```powershell
pip install TA-Lib-binary
```

> `requirements.txt` 의 플랫폼 마커(`platform_system == "Windows"`)가 자동으로 `TA-Lib-binary` 를 선택합니다.  
> 별도 C 라이브러리 설치 없이 바로 동작합니다.

#### 2) PyTorch — CUDA GPU 사용 시 pytorch.org 에서 버전 지정 설치

```powershell
# CUDA 12.1 예시 (GPU 드라이버에 맞는 CUDA 버전 확인 후 변경)
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# CPU 전용
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

> PyTorch를 먼저 수동 설치한 뒤 아래 단계로 나머지 패키지를 설치하세요.

#### 3) 나머지 패키지 설치

```powershell
pip install -r requirements.txt
```

---

### 🍎 macOS (Apple Silicon / Intel)

#### 1) TA-Lib — Homebrew로 C 라이브러리 선행 설치 필요

```bash
brew install ta-lib
```

#### 2) 패키지 일괄 설치

```bash
pip install -r requirements.txt
```

> Apple Silicon(M1/M2/M3)에서는 PyTorch MPS 백엔드가 자동으로 활성화됩니다.  
> `platform_utils.py` 가 디바이스를 자동 감지하므로 별도 설정 없이 사용할 수 있습니다.  
> 단, 소형 MLP 모델(RL Agent)의 경우 MPS 오버헤드가 CPU 대비 이득보다 클 수 있으므로 CPU 훈련을 권장합니다.

#### 3) (Linux) TA-Lib 시스템 라이브러리

```bash
# Debian/Ubuntu
sudo apt install libta-lib-dev
pip install -r requirements.txt
```

---

## 실행 흐름 (00 → 01 → 02 → 진화 파이프라인)

### Step 00 — 데이터 전처리 (최초 1회)

Raw OHLCV CSV에서 TA 지표를 계산하고 train-only 정규화 scaler를 저장합니다.

```bash
python scripts/00_prepare_data.py
python scripts/00_prepare_data.py --symbols BTC_USDT ETH_USDT SOL_USDT XRP_USDT
```

산출물: `data/processed/{symbol}_processed.csv`, `data/processed/scalers/`

---

### Step 01 — Base Expert 학습

Long / Short / Context 3개의 LSTM 신호 전문가를 학습합니다.

```bash
python scripts/01_train_base.py
python scripts/01_train_base.py --symbol ETH_USDT
```

산출물: `checkpoints/base_experts/{long,short,context}_expert.pth`

---

### Step 02 — Base Signal 추출

학습된 Expert로 전체 시계열에 대한 점수를 추론해 RL 학습용 신호 파일을 생성합니다.

```bash
python scripts/02_extract_signals.py
python scripts/02_extract_signals.py --symbol ETH_USDT
```

산출물: `data/signals/{symbol}_signals_log.csv`

---

### Step 03 — 진화 파이프라인 (★ 메인)

`run_evolution.py`가 훈련(04) → 백테스트(07) → 자동 폐기 → 다음 세대 이식을 자동 반복합니다.  
우승 모델은 `best_gen{N}_{tag}.zip` 형식으로 저장되어 태그(레버리지·프로파일 정보)가 보존됩니다.

```bash
# 시나리오 1: 1세대만 가볍게 (각 조합 10개, 전부 보존)
python run_evolution.py --target-generations 1 --count-per-task 10 --auto-discard-top 999

# 시나리오 2: 3세대 연속 진화 (1등만 살아남아 다음 세대 부모로 이식)
python run_evolution.py \
  --target-generations 3 \
  --auto-discard-top 1 \
  --leverages 1,2,3 \
  --profiles balanced,aggressive \
  --count-per-task 50

# 시나리오 3: 과거 모델에서 파인튜닝 시작
python run_evolution.py \
  --initial-parent checkpoints/rl_generations/gen1/best_gen1_lev2_bal_seed12345_001.zip \
  --target-generations 2
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--target-generations` | 1 | 진화시킬 세대 수 |
| `--leverages` | `1,3,5` | 레버리지 목록 (쉼표 구분) |
| `--profiles` | `stable,balanced,aggressive` | 하이퍼파라미터 프로파일 |
| `--count-per-task` | 10 | (레버리지 × 프로파일) 조합당 학습 모델 수 |
| `--jobs` | auto | 병렬 프로세스 수 (CPU 코어 자동 감지) |
| `--auto-discard-top` | 3 | 백테스트 상위 K개만 보존, 나머지 삭제 |
| `--initial-parent` | None | 1세대 시작 시 부모 가중치 경로 (파인튜닝) |
| `--mutation-scale-start` | 1.0 | 1세대 변이 폭 스케일 (0~1, 세대별 자동 감소) |

---

### Step 03b — Baby→Full 커리큘럼 파인튜닝 (선택)

Baby 환경 우승 가중치를 Full 환경(`LeverageTradingEnv`)으로 이식하여 심화 학습합니다.

```bash
python scripts/03b_finetune_full.py \
  --baby-model-path checkpoints/rl_generations/gen3/best_gen3_lev2_bal_seed12345_001.zip \
  --leverage 2 \
  --tuning-profile balanced \
  --data-path data/signals/BTC_USDT_signals_log.csv \
  --timesteps 500000 \
  --model-dir models/finetuned \
  --log-dir logs/finetune
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--baby-model-path` | (필수) | Baby 환경 우승 모델 경로 (.zip) |
| `--leverage` | 2 | 레버리지 배수 |
| `--tuning-profile` | balanced | stable / balanced / aggressive |
| `--timesteps` | 500000 | 총 학습 스텝 수 |
| `--eval-freq` | 10000 | 평가 주기 |

산출물: `model-dir/{tag}/final_model_{tag}.zip`

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
  --leverages 1,2 --profiles balanced,aggressive \
  --count-per-task 5 --jobs 3

# 단일 모델 백테스트 (전구간)
python scripts/05_backtest.py \
  --model-path checkpoints/rl_generations/gen1/lev2_bal_seed12345_001/final_model_lev2_bal_seed12345_001.zip \
  --tag lev2_bal_seed12345_001 \
  --data-path data/signals/BTC_USDT_signals_log.csv \
  --reports-dir reports/gen1

# 배치 백테스트
python scripts/07_backtest_batch.py \
  --model-dir checkpoints/rl_generations/gen1 \
  --data-path data/signals/BTC_USDT_signals_log.csv \
  --reports-dir reports/gen1

# Base Expert 시그널 검증
python scripts/tools/verify_base_signals.py \
  --split test --long-z 0.0 --short-z 0.0 --context-thresh 0.45

# 산출물 정리 (dry-run 먼저)
python scripts/tools/clear_artifacts.py --dry-run
python scripts/tools/clear_artifacts.py --targets logs,reports --yes
```

---

## 환경 구조

### BabyLeverageTradingEnv (Gen1~N 커리큘럼 훈련)

- **Action Space**: `Discrete(6)` — 0=홀드, 1=롱Full, 2=롱Half, 3=숏Full, 4=숏Half, 5=청산
- **Observation**: `Box(13,)` — long/short/context score, 포지션, 미실현 손익, 시간 피처, 레버리지, 홀드비율, DD
- **ActionMasking**: 포지션 없음 → 진입(1~4) 허용, 포지션 보유 → 청산(5)·홀드(0)만 허용
- **Train/Eval 분리**: `mode="train"` → 앞 70% | `mode="eval"` → 뒤 30%
- **Max Episode**: 20,000 스텝 (~70일, 5분봉 기준)
- **청산 벌점**: `LIQ_PENALTY = 100.0` (커리큘럼 완화)

### LeverageTradingEnv (Full 환경 / 03b_finetune_full.py)

Baby 환경과 동일한 Action Space / Observation 구조를 유지하며 다음이 추가됩니다:

- **ActionMasking**: Baby 환경과 동일 (`action_masks()` 메서드 포함)
- **본절컷(Breakeven Stop)**: 미실현 수익이 한번이라도 1% 이상 올랐다가 0 이하로 내려오면 강제 청산
- **승률 가중 보너스**: 자발적 청산 net_ret > 0 AND 에피소드 win_rate > 50% → 보상 × 1.2
- **청산 벌점**: `LIQ_PENALTY = 10000.0` (실전 수준)
- **복합 패널티**: 펀딩비, 드로우다운 방어, 롱/숏 편향 패널티, 손실 홀딩 시간 가중치
- **튜닝 프로파일**: stable / balanced / aggressive

---

## RL 학습 구조 (MaskablePPO)

모든 RL 학습(`03_train_rl.py`, `03b_finetune_full.py`)은 `sb3-contrib`의 `MaskablePPO`와 `ActionMasker`를 사용합니다.

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env): return env.action_masks()
env = Monitor(ActionMasker(LeverageTradingEnv(...), mask_fn))
model = MaskablePPO("MlpPolicy", env, ...)
```

이를 통해 포지션이 없는 상태에서 청산 액션(5)을 선택하거나, 포지션 보유 중 진입 액션(1~4)을 선택하는 **무효 액션이 구조적으로 차단**됩니다.

---

## 평가 파이프라인 (CustomEvalCallback)

`scripts/03_train_rl.py` 의 `CustomEvalCallback` 은 기본 `EvalCallback` 을 대체합니다.

### 승격 기준 (stability score)

```
score = mean_reward − (std_reward × 0.5) + (min_reward × 0.2)
```

단순 `max(mean_reward)` 대신 **변동성 패널티**(-) 와 **최악 케이스 보정**(+) 을 적용해,  
한 번 운 좋게 튄 불안정한 모델이 `best_model.zip` 을 덮어쓰는 것을 방지합니다.

### Tensorboard 지표

학습 중 아래 지표가 Tensorboard에 자동 기록됩니다.

| 키 | 설명 |
|----|------|
| `eval/stability_score` | stability score (승격 기준) |
| `eval/final_balance` | 에피소드 종료 잔고 평균 |
| `eval/win_rate` | 거래 승률(%) 평균 |
| `eval/total_trades` | 총 거래 횟수 평균 |

```bash
tensorboard --logdir logs/train
```

### eval_metrics.csv

모델 태그 폴더(`checkpoints/rl_generations/genN/<tag>/`) 에 매 평가 주기마다 누적 저장됩니다.

| 컬럼 | 설명 |
|------|------|
| `step` | 학습 타임스텝 |
| `score` | stability score |
| `mean_reward` / `std_reward` / `min_reward` | 에피소드 보상 통계 |
| `final_balance` | 에피소드 종료 잔고 평균 |
| `win_rate` | 거래 승률(%) 평균 |
| `total_trades` | 총 거래 횟수 평균 |
| `liquidation_count` | 강제청산 발생 에피소드 수 |

---

## 2단계 학습 구조 (Baby → Full)

1. **Gen1~N (Baby 환경)**: `BabyLeverageTradingEnv` 로 기본 롱/숏 방향 감지 습득
   - 복잡 패널티 없음, 커리큘럼 완화 보상, LIQ_PENALTY=100
2. **파인튜닝 (Full 환경)**: `03b_finetune_full.py` 로 `LeverageTradingEnv` 에 가중치 이식
   - LR=1e-5, ent_coef=0.003 (보수적, catastrophic forgetting 방지)
   - 본절컷·승률 보너스·복합 패널티 환경에서 추가 학습
3. **진화 파이프라인 내 파인튜닝** (`run_evolution.py`): `--initial-parent` 로 이전 세대 모델 투입 시
   - 변이 폭 자동 완화 (`mutation_scale × 0.5`)
   - `--mutation-scale-start` 로 세대별 변이 강도 조정 가능

---

## 우승 모델 태그 형식

```
lev{N}_{prof_code}_seed{seed}_{idx:03d}
예: lev2_bal_seed12345_001

우승 모델 저장: best_gen{N}_lev{N}_{prof_code}_seed{seed}_{idx:03d}.zip
부모 비교군:   parent_gen{N}_lev{N}_{prof_code}_seed{seed}_{idx:03d}.zip
```

| prof_code | 프로파일 |
|-----------|---------|
| `stb` | stable |
| `bal` | balanced |
| `agg` | aggressive |

---

## Base Expert 아키텍처 및 학습 설계

`src/models/base_models.py` 의 핵심 설계 결정 사항입니다.

### 모델 구조

| Expert | 입력 | 출력 | 손실 함수 |
|--------|------|------|-----------|
| `PriceActionExpert` (Long/Short) | OHLCV + 지표 시퀀스 | sigmoid 확률 | `FocalLoss(alpha=0.75, gamma=2)` |
| `ContextExpert` | OHLCV + 지표 시퀀스 | logit (raw) | `BCEWithLogitsLoss(pos_weight=...)` |

- LSTM 은닉층 이후 정규화는 시계열에 적합한 **`LayerNorm`** 을 사용합니다 (BatchNorm1d 미사용).
- `ContextExpert` 는 모델 내부 Sigmoid 없이 logit 을 출력하며, 추론 시 `torch.sigmoid()` 를 명시적으로 적용합니다.

### 정규화 방식 — prev_close 기반 Log Return

캔들 OHLC 4개 값을 각각 `shift(1)` 하는 대신, 모두 **직전 캔들의 종가(`prev_close`)** 를 기준으로 Log Return 을 계산합니다.

```
log_return_X = log(price_X / prev_close)   # X ∈ {open, high, low, close}
```

이를 통해 캔들 내부의 고가-저가-시가-종가 상대 관계(캔들 모양)가 보존됩니다.

### 라벨링 — Wick 기반 TP/SL 판별

5분봉 꼬리(Wick) 노이즈를 제거하기 위해 종가 대신 **고가/저가**로 TP·SL 터치를 판별합니다.

| 방향 | 익절(TP) 판별 | 손절(SL) 판별 |
|------|-------------|-------------|
| Long  | 미래 윈도우 **고가** ≥ TP 가격 | 미래 윈도우 **저가** ≤ SL 가격 |
| Short | 미래 윈도우 **저가** ≤ TP 가격 | 미래 윈도우 **고가** ≥ SL 가격 |

### 데이터셋 분할 및 샘플링

- **Train** : 클래스 불균형 완화를 위해 언더샘플링 (positive : negative ≈ 1 : 3) 적용
- **Val / Test** : 원본 인덱스를 그대로 사용 — 샘플링 없이 실제 분포 그대로 평가

### FocalLoss alpha

alpha 는 `0.75` 고정입니다. 1:3 언더샘플링 이후 데이터에는 raw pos_weight 기반의 동적 alpha 가 불필요하며, 과도한 alpha 가 all-ones 붕괴를 유발합니다.

---

## Notes

- `requirements.txt` 의 `torch` 버전을 변경하면 저장된 모델 직렬화 호환성이 깨질 수 있습니다.
- 세대별 로그는 `logs/train/gen{N}/` 에 `train.log` / `batch.log` / `backtest.log` / `rank.log` 로 분리 저장됩니다.
- `--auto-discard-top 999` 로 설정하면 폐기 없이 모든 산출물을 보존할 수 있습니다.
- **Apple Silicon(M1/M2/M3)**: RL Agent(소형 MLP)에서는 MPS 오버헤드 > 이득이므로 CPU 훈련을 권장합니다. `platform_utils.py` 가 디바이스를 자동 감지합니다.
- **Windows CUDA**: `requirements.txt` 의 `torch` 는 CPU/MPS fallback 용입니다. GPU 사용 시 `pytorch.org` 에서 CUDA 버전에 맞는 명령으로 먼저 설치하세요 (Setup 섹션 참고).
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` 설정이 자동 적용되어 멀티코어 병렬 훈련 효율이 최적화됩니다.
