# Legacy Scripts

초기(루트 기반) 학습/백테스트 스크립트를 이 폴더로 모았습니다.

## 목적
- 루트 디렉토리 정리
- v2/v3 파이프라인과 초기 실험 스크립트 분리
- 기존 실행 커맨드 호환 유지

## 포함된 스크립트
- `crypto_data_pipeline.py`
- `crypto_model_training.py`
- `crypto_backtester.py`
- `ensemble_trainer.py`
- `find_best_model.py`
- `prepare_backtest_cache.py`

## 호환성
루트에 동일 파일명의 래퍼가 남아 있어 기존처럼 실행 가능합니다.
예: `python crypto_model_training.py`

실제 구현은 `legacy/` 하위 파일을 수정하면 됩니다.
