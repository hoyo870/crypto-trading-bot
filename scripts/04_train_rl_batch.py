"""
Commander 병렬 일괄 훈련 오케스트레이터 (M1 Max 멀티코어 최적화)

여러 레버리지와 튜닝 프로파일 조합을 병렬로 큐(Queue)에 담아 실행합니다.
03_train_rl.py를 백그라운드 프로세스로 띄워 CPU 코어를 100% 활용하며,
훈련이 모두 끝나면 05_backtest.py를 자동으로 호출하여 최종 리포트를 뽑아냅니다.
"""

import os
import sys
import time
import argparse
import subprocess
from collections import deque
from itertools import product

# ── 경로 설정 ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "03_train_rl.py")
BACKTEST_SCRIPT = os.path.join(SCRIPT_DIR, "05_backtest.py")


def _default_jobs() -> int:
    cpu_count = os.cpu_count() or 2
    # Leave one core for OS/IO and keep minimum 1 worker.
    return max(1, cpu_count - 1)


def _safe_stop_processes(processes):
    for proc, *_ in processes:
        if proc.poll() is None:
            proc.terminate()

    deadline = time.time() + 10
    for proc, *_ in processes:
        if proc.poll() is None:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()


def _repo_abs(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(ROOT_DIR, path_value)

def run_parallel_orchestrator(args):
    data_path = _repo_abs(args.data_path)
    model_dir = _repo_abs(args.model_dir)
    log_dir = _repo_abs(args.log_dir)
    backtest_data_path = _repo_abs(args.backtest_data_path)
    reports_dir = _repo_abs(args.reports_dir)
    best_output = _repo_abs(args.best_output)
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONIOENCODING", "utf-8")

    # 입력받은 옵션을 리스트로 변환
    leverages = [int(x.strip()) for x in args.leverages.split(",")]
    profiles = [x.strip() for x in args.profiles.split(",")]
    
    # 훈련 작업 큐(Queue) 생성 (예: 1x-stable, 1x-balanced, 3x-stable 등 모든 조합)
    tasks = deque(list(product(leverages, profiles)))
    total_tasks = len(tasks)
    active_procs = []
    completed_tags = []
    logs_dir = os.path.join(SCRIPT_DIR, "logs", "batch_train")
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"\n{'='*65}")
    print(f"🚀 [M1 Max 병렬 훈련 사령탑 가동]")
    print(f"총 {total_tasks}개의 훈련 그룹이 큐(Queue)에 등록되었습니다.")
    print(f"그룹당 시드(모델) 수: {args.count_per_task}개 | 동시 실행(코어) 수: {args.jobs}개")
    print(f"{'='*65}\n")
    
    start_time = time.time()
    completed_tasks = 0

    # 큐에 남은 작업이 있거나, 아직 실행 중인 프로세스가 있으면 루프 유지
    while tasks or active_procs:
        
        # 슬롯이 비어있으면 큐에서 작업을 꺼내어 새 프로세스 투입
        while tasks and len(active_procs) < args.jobs:
            lev, prof = tasks.popleft()
            base_tag = f"lev{lev}_{args.tag_prefix}_{prof}"
            expected_tags = [base_tag] if args.count_per_task == 1 else [
                f"{base_tag}_{i + 1:03d}" for i in range(args.count_per_task)
            ]
            
            cmd = [
                sys.executable,
                TRAIN_SCRIPT,
                "--leverage", str(lev),
                "--tuning-profile", prof,
                "--count", str(args.count_per_task),
                "--tag", base_tag,
                "--timesteps", str(args.timesteps),
                "--eval-freq", str(args.eval_freq),
                "--patience", str(args.patience),
                "--data-path", data_path,
                "--model-dir", model_dir,
                "--log-dir", log_dir,
            ]
            
            if args.load_model:
                cmd.extend(["--load-model", args.load_model])
                
            log_path = os.path.join(logs_dir, f"train_lev{lev}_{prof}.log")
            print(f"▶️ [START] 레버리지: {lev}x | 프로파일: {prof:<10} | 시드 투입: {args.count_per_task}개")
            print(f"    ↳ 로그 파일: {log_path}")
            
            # 백그라운드 프로세스 실행
            log_file = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=SCRIPT_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=child_env,
            )
            active_procs.append((proc, lev, prof, log_file, expected_tags))
        
        # 현재 돌고 있는 프로세스들의 상태 모니터링
        for proc, lev, prof, log_file, expected_tags in active_procs[:]:
            ret = proc.poll()
            
            # 프로세스가 종료되었으면 (ret is not None)
            if ret is not None:
                log_file.close()
                active_procs.remove((proc, lev, prof, log_file, expected_tags))
                completed_tasks += 1
                
                # 에러 발생 시 즉시 중단 안전장치
                if ret != 0:
                    print(f"\n[ERROR] ❌ {lev}x ({prof}) 훈련 그룹에서 치명적 에러 발생! (Exit code: {ret})")
                    print("[INFO] 메모리 안전을 위해 실행 중인 나머지 프로세스를 모두 강제 종료합니다.")
                    _safe_stop_processes(active_procs)
                    for _, _, _, lf, _ in active_procs:
                        try:
                            lf.close()
                        except Exception:
                            pass
                    sys.exit(ret)
                else:
                    completed_tags.extend(expected_tags)
                    print(f"✅ [DONE] 레버리지: {lev}x | 프로파일: {prof:<10} (완료: {completed_tasks}/{total_tasks})")
        
        # 무한 루프 과부하 방지
        time.sleep(1) 

    total_elapsed = time.time() - start_time
    hours, rem = divmod(total_elapsed, 3600)
    mins, secs = divmod(rem, 60)
    
    print(f"\n{'='*65}")
    print(f"🎉 모든 파이프라인 훈련이 완벽하게 종료되었습니다!")
    print(f"⏱️ 총 소요 시간: {int(hours)}시간 {int(mins)}분 {int(secs)}초")
    print(f"{'='*65}\n")

    # ── 훈련 완료 후 자동 백테스트 체인 ──
    if args.run_backtest:
        print(f"📈 [INFO] 05_backtest.py를 호출하여 이번 세대의 베스트 모델을 선출합니다...\n")
        bt_cmd = [
            sys.executable,
            BACKTEST_SCRIPT,
            "--model-dir", model_dir,
            "--data-path", backtest_data_path,
            "--reports-dir", reports_dir,
            "--best-output", best_output,
        ]
        if completed_tags:
            bt_cmd.extend(["--tags", ",".join(completed_tags)])
        
        subprocess.run(bt_cmd, cwd=SCRIPT_DIR, check=True, env=child_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="병렬 RL 훈련 오케스트레이터")
    
    # 작업 구성 옵션
    parser.add_argument("--leverages", type=str, default="1,3,5", help="쉼표로 구분된 레버리지 목록 (예: 1,3,5)")
    parser.add_argument("--profiles", type=str, default="stable,balanced,aggressive", help="쉼표로 구분된 튜닝 프로파일")
    parser.add_argument("--count-per-task", type=int, default=10, help="각 그룹(조합)당 훈련할 모델 수 (기본: 10)")
    parser.add_argument("--tag-prefix", type=str, default="batch", help="생성 모델 태그 접두사")
    
    # 하드웨어 최적화 옵션
    parser.add_argument("--jobs", type=int, default=_default_jobs(), help="동시 실행할 병렬 프로세스 수 (기본: CPU-1)")

    # 학습 파라미터 전달 옵션
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="03_train_rl.py로 전달할 timesteps")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="03_train_rl.py로 전달할 eval-freq")
    parser.add_argument("--patience", type=int, default=30, help="03_train_rl.py로 전달할 patience")
    parser.add_argument("--data-path", type=str, default=os.path.join("data", "signals", "base_signals_log.csv"),
                        help="03_train_rl.py로 전달할 입력 데이터 경로")
    parser.add_argument("--model-dir", type=str, default=os.path.join("checkpoints", "rl_generations"),
                        help="03_train_rl.py 및 05_backtest.py 모델 경로")
    parser.add_argument("--log-dir", type=str, default=os.path.join("logs", "tensorboard"),
                        help="03_train_rl.py로 전달할 텐서보드 로그 경로")
    
    # 파이프라인 연결 옵션
    parser.add_argument("--run-backtest", action="store_true", default=True, help="훈련 종료 후 자동 일괄 백테스트 실행")
    parser.add_argument("--no-backtest", dest="run_backtest", action="store_false", help="자동 백테스트 끄기")
    parser.add_argument("--load-model", type=str, default=None, help="(커리큘럼 학습용) 파인튜닝할 부모 모델 zip 경로")
    parser.add_argument("--backtest-data-path", type=str, default=os.path.join("data", "signals", "base_signals_log.csv"),
                        help="05_backtest.py로 전달할 입력 데이터 경로")
    parser.add_argument("--reports-dir", type=str, default=os.path.join("reports", "batch"),
                        help="05_backtest.py 리포트 출력 경로")
    parser.add_argument("--best-output", type=str, default=os.path.join("reports", "batch", "best_by_leverage.csv"),
                        help="05_backtest.py 베스트 CSV 출력 경로")
    
    args = parser.parse_args()
    run_parallel_orchestrator(args)