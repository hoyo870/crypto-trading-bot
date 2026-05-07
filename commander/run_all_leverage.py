import os
import sys
import subprocess
import random

# 현재 스크립트가 있는 디렉토리
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_sequential_trainings():
    # 실행할 (count, leverage) 작업 목록
    tasks = [
        (7, 1),
        (8, 3),
        (10, 5)
    ]

    print(f"\n{'='*60}")
    print("🚀 [배치 자동화] 레버리지별 순차 학습 파이프라인 가동 시작")
    print(f"{'='*60}")

    for count, leverage in tasks:
        print(f"\n{'*'*50}")
        print(f"▶️ 현재 실행 중: 레버리지 {leverage}x (시드 {count}개)")
        print(f"{'*'*50}\n")

        # 터미널 명령어 구성
        cmd = [
            sys.executable, 
            os.path.join(BASE_DIR, "run_train.py"),
            "--count", str(count),
            "--leverage", str(leverage),
            "--seeds", ",".join(str(random.randint(0, 10000)) for _ in range(count))
        ]

        # 명령어 실행 (터미널 출력은 그대로 화면에 보여줌)
        result = subprocess.run(cmd, cwd=BASE_DIR)

        # 에러 체크: 프로세스가 0(성공)이 아닌 코드로 종료되면 전체 스크립트 중단
        if result.returncode != 0:
            print(f"\n[ERROR] ❌ 레버리지 {leverage}x 훈련 중 치명적인 에러 발생! (Exit code: {result.returncode})")
            print("[INFO] 안전을 위해 후속 훈련을 모두 중단합니다.")
            sys.exit(result.returncode)
        
        print(f"\n[SUCCESS] ✅ 레버리지 {leverage}x 훈련 무사히 완료!")

    print(f"\n{'='*60}")
    print("🎉 모든 순차적 학습 (1x, 3x, 5x)이 완벽하게 종료되었습니다!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_sequential_trainings()