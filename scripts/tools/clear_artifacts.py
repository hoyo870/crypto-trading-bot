"""
Commander 실험 산출물 정리 스크립트.

기본 동작:
- commander/logs 하위 내용 삭제
- commander/reports 하위 내용 삭제
- root/models/commander 하위 내용 삭제
- 빈 디렉토리 구조 재생성

예시:
  python clear_artifacts.py --dry-run
  python clear_artifacts.py --yes
  python clear_artifacts.py --targets logs,reports
"""
import argparse
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


def _collect_targets(targets):
    entries = []

    if "logs" in targets:
        entries.extend([
            BASE_DIR / "logs" / "train",
            BASE_DIR / "logs" / "backtest",
        ])

    if "reports" in targets:
        entries.append(BASE_DIR / "reports")

    if "models" in targets:
        entries.extend([
            ROOT_DIR / "models" / "commander" / "candidates",
            ROOT_DIR / "models" / "commander" / "runs",
        ])

    return entries


def _iter_children(directory):
    if not directory.exists():
        return []
    return sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))


def _remove_path(path, dry_run):
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _recreate_layout(targets):
    if "logs" in targets:
        (BASE_DIR / "logs" / "train").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "logs" / "backtest").mkdir(parents=True, exist_ok=True)

    if "reports" in targets:
        (BASE_DIR / "reports").mkdir(parents=True, exist_ok=True)

    if "models" in targets:
        (ROOT_DIR / "models" / "commander" / "candidates").mkdir(parents=True, exist_ok=True)
        (ROOT_DIR / "models" / "commander" / "runs").mkdir(parents=True, exist_ok=True)


def clear_artifacts(targets, dry_run=False):
    target_dirs = _collect_targets(targets)
    planned = []

    for directory in target_dirs:
        for child in _iter_children(directory):
            planned.append(child)

    print("[INFO] 정리 대상")
    for name in sorted(targets):
        print(f"  - {name}")

    if not planned:
        print("[INFO] 삭제할 산출물이 없습니다.")
        _recreate_layout(targets)
        return 0

    print(f"[INFO] 삭제 예정 항목 수: {len(planned)}")
    for path in planned:
        print(f"  - {path}")

    for path in planned:
        _remove_path(path, dry_run=dry_run)

    _recreate_layout(targets)

    if dry_run:
        print("[DONE] dry-run 완료. 실제 삭제는 수행하지 않았습니다.")
    else:
        print("[DONE] commander 산출물 정리 완료.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Commander 산출물(logs/reports/models) 일괄 정리")
    parser.add_argument(
        "--targets",
        type=str,
        default="logs,reports,models",
        help="정리 대상 목록. logs,reports,models 중 쉼표로 지정",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="삭제하지 않고 대상만 출력",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="확인 프롬프트 없이 즉시 실행",
    )
    args = parser.parse_args()

    valid_targets = {"logs", "reports", "models"}
    targets = {token.strip() for token in args.targets.split(",") if token.strip()}

    unknown = sorted(targets - valid_targets)
    if unknown:
        parser.error(f"알 수 없는 targets 값: {unknown}")

    if not targets:
        parser.error("최소 하나 이상의 target이 필요합니다.")

    if not args.dry_run and not args.yes:
        print("[WARN] commander 산출물을 전량 삭제합니다.")
        answer = input("계속하려면 'yes' 를 입력하세요: ").strip().lower()
        if answer != "yes":
            print("[INFO] 작업을 취소했습니다.")
            return 1

    return clear_artifacts(targets=targets, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())