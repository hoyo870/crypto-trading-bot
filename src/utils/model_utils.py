"""
src/utils/model_utils.py
모델 파일 경로 탐색 등 공통 유틸리티.

scripts/05_backtest.py 와 scripts/07_backtest_batch.py 에서 중복 정의되던
_resolve_model_path 를 이곳으로 통합.
"""

import os


def resolve_model_path(tag: str, model_dir: str) -> str | None:
    """
    태그 → 모델 .zip 경로 자동 탐색.

    탐색 순서:
      1) <model_dir>/<tag>/ 폴더 안의 *final*.zip 또는 *best*.zip
      2) <model_dir>/<tag>.zip 직접 경로
    반환: 절대 경로 문자열, 없으면 None.
    """
    clean = tag[:-4] if tag.endswith(".zip") else tag
    folder = os.path.join(model_dir, clean)
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.endswith(".zip") and ("final" in f or "best" in f):
                return os.path.join(folder, f)
    direct = os.path.join(model_dir, f"{clean}.zip")
    if os.path.exists(direct):
        return direct
    return None
