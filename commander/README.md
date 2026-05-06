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

R&D 옵션 (run_train.py):
- `--split-mode holdout` + `--train-ratio` + `--eval-ratio`:
	학습/평가 구간 분리
- `--top-k N`:
	학습 완료 후 candidates 상위 N개만 유지

예시:
```bash
python commander/run_train.py \
	--count 5 --leverage 3 --timesteps 200000 \
	--split-mode holdout --train-ratio 0.7 --eval-ratio 0.2 \
	--train-ep-steps 20000 --eval-window 20000 \
	--top-k 3
```