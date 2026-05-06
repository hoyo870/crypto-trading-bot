# Crypto Bot

현재 저장소는 `commander`(현행) + `legacy`(구버전) 구조로 운영됩니다.

## 폴더 구조

- `commander/`: 현재 사용 중인 학습/백테스트 코드
- `legacy/`: 이전 실험 코드 및 레거시 스크립트
- `data/`: 공용 데이터 저장소
  - `data/commander/`: commander 입력/파생 데이터
  - `data/scalers/`: 스케일러
- `models/`: 공용 모델 저장소
  - `models/commander/`: commander RL 모델
  - `models/legacy/`: 레거시 모델

## Commander 기본 실행 순서

1. Base 모델 학습

```bash
python commander/train_base_models.py
```

2. Base 시그널 생성

```bash
python commander/validate_base_signals.py
```

3. RL 학습

```bash
python commander/run_train.py --count 1 --leverage 3 --patience 30
```

4. 백테스트

```bash
python commander/run_backtest.py --leverage 3
```

## 모델 파일 규칙

자동 생성 태그 형식:

- `lev{레버리지}_seed{시드}_NNN`

예시:

- `lev1_seed42_001`
- `lev3_seed42_001`

## 참고

- `commander`의 기본 데이터 경로는 `data/commander/base_signals_log.csv` 입니다.
- `commander`의 기본 모델 경로는 `models/commander/` 입니다.