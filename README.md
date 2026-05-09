# Crypto Trading Bot

강화학습 기반 암호화폐 트레이딩 연구/실험 저장소입니다.

현재 파이프라인은 scripts 폴더(01~05)와 src 폴더를 기준으로 동작합니다.

## Repository Layout

- scripts/: 실행 엔트리 포인트 (01~05)
- src/: 모델/환경 핵심 코드
- checkpoints/: 학습 결과물 (base_experts, rl_generations)
- data/: raw/processed/signals 데이터
- legacy/: 과거 실험 코드(참고용)
- requirements.txt: 의존성 목록

## Setup

### 1) Python 가상환경 생성

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS (zsh/bash):

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) TA-Lib 안내 (macOS)

requirements.txt에는 OS 분기 처리가 되어 있습니다.

- Windows: TA-Lib-binary 설치
- macOS/Linux: TA-Lib 설치

macOS에서 TA-Lib 설치가 실패하면 아래를 먼저 실행하세요.

```bash
brew install ta-lib
pip install -r requirements.txt
```

## Smoke Scenario (01 -> 02 -> 03 -> 05)

아래는 현재 구조 기준 최소 실행 예시입니다.

### 01) Base Expert 학습

```bash
python scripts/01_train_base.py --data-path data/processed/BTC_USDT_processed.csv
```

산출물:

- checkpoints/base_experts/long_expert.pth
- checkpoints/base_experts/short_expert.pth
- checkpoints/base_experts/context_expert.pth

### 02) Base Signal 생성

02 스크립트는 processed 파일명 규칙에서 raw 파일 경로를 유도합니다.
예: data/processed/BTC_USDT_processed.csv -> data/processed/BTC_USDT_5m_raw.csv

```bash
python scripts/02_extract_signals.py \
  --data-path data/processed/BTC_USDT_processed.csv \
  --output-filename base_signals_log.csv
```

산출물:

- data/signals/base_signals_log.csv

### 03) RL 학습

```bash
python scripts/03_train_rl.py \
  --data-path data/signals/base_signals_log.csv \
  --leverage 2 \
  --count 1 \
  --timesteps 3000000
```

산출물:

- checkpoints/rl_generations/<tag>/best_model.zip
- checkpoints/rl_generations/<tag>/final_model_<tag>.zip

### 05) 백테스트

```bash
python scripts/05_backtest.py \
  --model-dir checkpoints/rl_generations \
  --data-path data/signals/base_signals_log.csv \
  --reports-dir reports
```

산출물:

- reports/rl_backtest_summary_<tag>.json
- reports/rl_backtest_chart_<tag>.png
- reports/best_by_leverage.csv

## Notes

- RL 직렬화 호환성을 위해 requirements.txt의 torch/stable-baselines3 버전을 유지하는 것을 권장합니다.
- 대량 실행은 scripts/04_train_rl_batch.py를 사용하세요.
