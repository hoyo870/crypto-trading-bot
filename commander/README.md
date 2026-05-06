# Commander

통합된 RL 트레이딩 파이프라인입니다.

- base expert 학습: train_base_models.py
- base signal 생성: validate_base_signals.py
- commander 학습: train_rl_commander.py
- commander 배치 학습: run_train.py
- commander 백테스트: backtest_rl_commander.py
- commander 배치 백테스트: run_backtest.py

모델 저장 경로 기본값:
- root/models/commander

데이터 경로 기본값:
- root/data/commander/base_signals_log.csv

자동 모델 태그 규칙:
- lev{레버리지}_seed{시드}_NNN