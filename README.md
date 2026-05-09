# Crypto Bot

강화학습 기반 암호화폐 트레이딩 연구/실험 저장소입니다.

핵심 목적은 다음과 같습니다.

1. Base Expert 모델(롱/숏/컨텍스트)로 시계열 점수를 생성한다.
2. 점수 로그를 입력으로 Commander RL 정책(PPO)을 학습한다.
3. 대량 후보를 백테스트하고 성과 기준으로 정제한다.
4. 프로파일별 상위 모델을 세대(Gen1, Gen2) 단위로 운영한다.

현재 실사용 파이프라인은 commander 폴더에 있습니다.

## Repository Layout

- commander/: 현행 파이프라인(학습/백테스트/오케스트레이션)
- legacy/: 과거 실험 코드(참고용)
- data/: 입력 데이터 및 파생 로그
- models/: 학습 결과 모델 아티팩트
- requirements.txt: 의존성 목록

## End-to-End Pipeline

### 1) Base Expert 학습

롱/숏/컨텍스트 전문가 모델을 학습하여 저장합니다.

```bash
python commander/train_base_models.py
```

출력:

- models/commander/base/long_expert.pth
- models/commander/base/short_expert.pth
- models/commander/base/context_expert.pth

### 2) Base Signal 생성

전문가 모델 추론으로 시점별 점수 로그를 생성합니다.

```bash
python commander/validate_base_signals.py
```

출력:

- data/commander/base_signals_log.csv
- data/commander/base_signals_threshold_scan.csv

### 3) Commander RL 학습

배치 학습(다중 모델) 또는 단일 학습을 실행합니다.

```bash
python commander/run_train.py --count 3 --leverage 1 --tuning-profile balanced
```

출력:

- models/commander/runs/<tag>/...
- models/commander/candidates/<tag>.zip
- commander/logs/train/*.log

### 4) 백테스트

후보 모델을 일괄 백테스트하고 요약 리포트를 생성합니다.

```bash
python commander/run_backtest.py --source candidates --workers 3
```

출력:

- commander/reports/rl_backtest_summary_<tag>.json
- commander/reports/rl_backtest_report_<tag>.txt
- commander/reports/rl_backtest_balance_<tag>.png
- commander/logs/backtest/*.log

## Generation 운영 (Gen1 -> Gen2)

본 프로젝트는 후보 정제 후 세대 기반으로 모델을 운용합니다.

- Gen1 저장 위치: models/commander/gen1/
- 메타 파일: models/commander/gen1/gen1_meta.json
- Gen2 학습 시 Gen1을 초기 가중치로 로드 가능

예시:

```bash
python commander/run_all_leverage.py \
  --gen1-meta ../models/commander/gen1/gen1_meta.json \
  --count-per-task 33 \
  --parallel 3
```

## 주요 실행 스크립트

- commander/train_base_models.py: Base Expert 학습
- commander/validate_base_signals.py: 점수 로그 생성 및 임계치 스캔
- commander/train_rl_commander.py: 단일 RL 학습 엔진
- commander/run_train.py: RL 배치 학습 래퍼
- commander/backtest_rl_commander.py: 단일 백테스트
- commander/run_backtest.py: 백테스트 배치 실행
- commander/run_all_leverage.py: 병렬 학습 + 학습 후 자동 백테스트 오케스트레이터

## 모델 태그 규칙

현재 태그는 프로파일/시드/시각 정보를 포함합니다.

- lev{leverage}_{profileCode}_seed{seed}_t{HHMMSS}_r{nonce}

예시:

- lev1_bal_seed6726_t034738_r131
- lev1_agg_seed8624_t034738_r911_012

profileCode:

- stb: stable
- bal: balanced
- agg: aggressive

## 데이터/경로 기본값

- 입력 데이터 기본값: data/commander/base_signals_log.csv
- 모델 루트 기본값: models/commander/
- 리포트 기본값: commander/reports/

## 운영 가이드

1. base_signals_log.csv 갱신 후 RL 학습을 시작합니다.
2. candidates를 백테스트하여 성과 기준으로 정제합니다.
3. 프로파일별 최종 모델을 Gen1로 보관합니다.
4. Gen1을 초기 가중치로 Gen2 파인튜닝을 진행합니다.

자세한 실행 옵션과 실험 파라미터는 commander/README.md를 참고하세요.

## 운영용 치트시트

아래 명령은 루트 디렉토리(crypto_bot)에서 실행하는 기준입니다.

### 환경 활성화

```bash
conda activate cryptobot
```

### Base -> Signal -> RL -> Backtest (기본 1사이클)

```bash
python commander/train_base_models.py
python commander/validate_base_signals.py
python commander/run_train.py --count 3 --leverage 1 --tuning-profile balanced
python commander/run_backtest.py --source candidates --workers 3 --tuning-profile balanced
```

### 특정 태그만 백테스트

```bash
python commander/run_backtest.py \
  --source candidates \
  --tags lev1_bal_seed6726_t034738_r131_033,lev1_agg_seed8624_t034738_r911_012
```

### 레버리지/프로파일 병렬 학습 + 자동 백테스트

```bash
python commander/run_all_leverage.py \
  --parallel 3 \
  --leverages 1 \
  --profiles stable,balanced,aggressive \
  --count-per-task 33
```

### Gen1 기반 Gen2 파인튜닝

```bash
python commander/run_all_leverage.py \
  --gen1-meta models/commander/gen1/gen1_meta.json \
  --parallel 3 \
  --count-per-task 33
```

### 프로파일별 추천 템플릿

stable (낙폭 억제 우선):

```bash
python commander/run_train.py \
  --count 3 \
  --leverage 1 \
  --tuning-profile stable \
  --timesteps 5000000 \
  --patience 40 \
  --split-mode holdout
```

balanced (기본 권장):

```bash
python commander/run_train.py \
  --count 3 \
  --leverage 1 \
  --tuning-profile balanced \
  --timesteps 5000000 \
  --patience 30 \
  --split-mode holdout
```

aggressive (수익 상한 탐색):

```bash
python commander/run_train.py \
  --count 3 \
  --leverage 1 \
  --tuning-profile aggressive \
  --timesteps 5000000 \
  --patience 20 \
  --split-mode holdout
```

프로파일 3종 동시 학습:

```bash
python commander/run_all_leverage.py \
  --parallel 3 \
  --leverages 1 \
  --profiles stable,balanced,aggressive \
  --count-per-task 33
```

### 자주 보는 파일

- commander/reports/rl_backtest_summary_<tag>.json
- models/commander/gen1/gen1_meta.json
- commander/logs/train/
- commander/logs/backtest/