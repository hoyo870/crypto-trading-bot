# Commander

Commander는 Base Expert 신호를 입력으로 사용하는 PPO 기반 강화학습 트레이딩 파이프라인입니다.

## 설계 의도

1. 고정 규칙 기반이 아니라, 시장 국면 변화에 적응 가능한 정책 학습
2. 대량 후보를 자동 생성/평가하여 재현 가능한 선별 체계 구축
3. 프로파일(stable/balanced/aggressive)별 독립 운용
4. Gen1, Gen2와 같은 세대형 파인튜닝 운영

## 구성 요소

- train_base_models.py: 롱/숏/컨텍스트 Expert 학습
- validate_base_signals.py: Expert 점수 추론 및 로그 생성
- train_rl_commander.py: 단일 RL 모델 학습 엔진(PPO)
- run_train.py: RL 배치 학습 래퍼
- backtest_rl_commander.py: 단일 모델 백테스트
- run_backtest.py: 다중 태그 병렬 백테스트
- run_all_leverage.py: 병렬 학습 오케스트레이션 + 학습 완료 후 자동 백테스트

## 데이터 흐름

1. 원본/가공 데이터 입력
2. Base Expert 학습
3. 시점별 점수(long_score, short_score, context_score) 생성
4. RL 학습 데이터(base_signals_log.csv)로 정책 학습
5. 후보 모델 백테스트 및 성능 리포트 생성
6. 필터링 후 Gen1 보관, Gen2 파인튜닝에 재사용

## 기본 경로

- 데이터: root/data/commander/base_signals_log.csv
- 모델 루트: root/models/commander
- 실행 로그: commander/logs/
- 백테스트 리포트: commander/reports/

## 빠른 시작

### 1) Base Expert 학습

```bash
python train_base_models.py
```

### 2) Signal 로그 생성

```bash
python validate_base_signals.py
```

### 3) RL 배치 학습

```bash
python run_train.py \
	--count 3 \
	--leverage 1 \
	--timesteps 5000000 \
	--tuning-profile balanced \
	--split-mode holdout
```

### 4) 일괄 백테스트

```bash
python run_backtest.py \
	--source candidates \
	--workers 3 \
	--tuning-profile balanced
```

## 주요 옵션

### run_train.py

- --count: 학습할 모델 수
- --leverage: 레버리지 배수
- --tuning-profile: stable | balanced | aggressive
- --split-mode: none | holdout
- --train-ratio, --eval-ratio: holdout 분할 비율
- --top-k: 학습 후 candidates 유지 개수
- --load-model: 기존 모델 zip을 초기 가중치로 로드(파인튜닝)
- --tag: 태그 접두어 지정
- --seeds: 쉼표 구분 시드 목록 고정

### run_backtest.py

- --source: candidates | runs
- --tags: 특정 태그만 백테스트
- --workers: 병렬 스레드 수
- --pick-best-per-leverage: 레버리지별 베스트 자동 추출
- --best-metric: score | total_return_pct | sharpe_ratio | mdd_pct

### run_all_leverage.py

- --parallel: 동시 학습 프로세스 수
- --leverages: 쉼표 구분 레버리지 목록
- --profiles: stable,balanced,aggressive
- --count-per-task: (레버리지,프로파일) 조합당 학습 모델 수
- --gen1-meta: Gen1 메타 JSON 경로(프로파일별 초기 모델 자동 로드)
- --run-backtest / --no-run-backtest: 학습 후 자동 백테스트 on/off

## 태그 규칙

run_all_leverage.py 기반 태그:

- lev{leverage}_{profileCode}_seed{seed}_t{HHMMSS}_r{nonce}

count>1일 때는 접미사가 추가됩니다.

- lev1_bal_seed6726_t034738_r131_001
- lev1_bal_seed6726_t034738_r131_002

profileCode:

- stb: stable
- bal: balanced
- agg: aggressive

## 세대형 운영 (Gen1 -> Gen2)

Gen1 후보는 아래 경로에 보관합니다.

- models/commander/gen1/
- models/commander/gen1/gen1_meta.json

Gen2 학습 시 아래처럼 Gen1 메타를 연결하면 프로파일별로 자동 파인튜닝됩니다.

```bash
python run_all_leverage.py \
	--gen1-meta ../models/commander/gen1/gen1_meta.json \
	--parallel 3 \
	--count-per-task 33
```

## 출력 아티팩트

- models/commander/runs/<tag>/: 학습 체크포인트, eval 로그
- models/commander/candidates/<tag>.zip: 승격된 후보 모델
- commander/logs/train/*.log: 학습 로그
- commander/logs/backtest/*.log: 백테스트 로그
- commander/reports/rl_backtest_summary_<tag>.json: 핵심 지표 요약

## 지표 해석 기준(권장)

- total_return_pct: 누적 수익률
- mdd_pct: 최대 낙폭(절대값이 작을수록 안정적)
- sharpe_ratio: 위험 대비 수익 효율
- total_trades: 거래 빈도(과소적합/과최적화 점검 지표)

프로파일별 선별 시 수익률 단독보다 MDD, Sharpe, 거래 수를 함께 보는 것을 권장합니다.

## 운영용 치트시트

아래 명령은 commander 디렉토리에서 실행하는 기준입니다.

### 1) Base Expert 학습

```bash
python train_base_models.py
```

### 2) Signal 로그 생성

```bash
python validate_base_signals.py
```

### 3) RL 배치 학습 (프로파일 지정)

```bash
python run_train.py \
	--count 3 \
	--leverage 1 \
	--tuning-profile balanced \
	--split-mode holdout
```

### 4) 후보 백테스트

```bash
python run_backtest.py \
	--source candidates \
	--workers 3 \
	--tuning-profile balanced
```

### 5) 베스트 후보 CSV 생성

```bash
python run_backtest.py \
	--source candidates \
	--pick-best-per-leverage \
	--best-metric score
```

### 6) 병렬 오케스트레이션 (학습 + 자동 백테스트)

```bash
python run_all_leverage.py \
	--parallel 3 \
	--leverages 1 \
	--profiles stable,balanced,aggressive \
	--count-per-task 33
```

### 7) Gen1 기반 Gen2 파인튜닝

```bash
python run_all_leverage.py \
	--gen1-meta ../models/commander/gen1/gen1_meta.json \
	--parallel 3 \
	--count-per-task 33
```

### 8) 프로파일별 추천 파라미터 템플릿

stable (낙폭 억제 우선):

```bash
python run_train.py \
	--count 3 \
	--leverage 1 \
	--tuning-profile stable \
	--timesteps 5000000 \
	--patience 40 \
	--split-mode holdout
```

balanced (기본 권장):

```bash
python run_train.py \
	--count 3 \
	--leverage 1 \
	--tuning-profile balanced \
	--timesteps 5000000 \
	--patience 30 \
	--split-mode holdout
```

aggressive (수익 상한 탐색):

```bash
python run_train.py \
	--count 3 \
	--leverage 1 \
	--tuning-profile aggressive \
	--timesteps 5000000 \
	--patience 20 \
	--split-mode holdout
```

프로파일 3종 동시 학습:

```bash
python run_all_leverage.py \
	--parallel 3 \
	--leverages 1 \
	--profiles stable,balanced,aggressive \
	--count-per-task 33
```

### 9) 실패 시 빠른 점검

```bash
python run_train.py --help
python run_backtest.py --help
python run_all_leverage.py --help
```