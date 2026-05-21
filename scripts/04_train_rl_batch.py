"""
Commander 병렬 일괄 훈련 오케스트레이터 (멀티코어 최적화)

오직 03_train_rl.py를 백그라운드 프로세스로 띄워 CPU 코어를 100% 활용하는
'병렬 훈련 큐(Queue) 관리' 역할만 수행합니다.
(백테스트 기능은 05_backtest.py로 완전히 분리되었습니다.)
"""

import os
import sys
import time
import argparse
import subprocess
from collections import deque
from itertools import product
import logging

# 경로 설정 후 platform_utils 임포트
# ── 경로 설정 ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "03_train_rl.py")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.platform_utils import get_optimal_jobs

# ── 로깅 설정 ─────────────────────────────────────────────────────────────
os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "orchestrator.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Batch")

# ── 세대별 로그 파일 추가 (CUSTOM_LOG_DIR 환경변수 세팅 시 batch.log 병행 기록) ─────────────
_custom_log_dir = os.environ.get("CUSTOM_LOG_DIR")
if _custom_log_dir:
    os.makedirs(_custom_log_dir, exist_ok=True)
    logging.getLogger().addHandler(
        logging.FileHandler(os.path.join(_custom_log_dir, "batch.log"), encoding='utf-8')
    )


def run_parallel_orchestrator(args):
    leverages = [int(x.strip()) for x in args.leverages.split(",")]
    profiles = [x.strip() for x in args.profiles.split(",")]
    
    # 훈련 작업 큐(Queue) 생성
    tasks = deque(list(product(leverages, profiles)))
    total_tasks = len(tasks)
    active_procs = []
    
    logger.info("")
    logger.info(f"{'='*65}")
    logger.info(f"🚀 [병렬 훈련 오케스트레이터 가동]")
    logger.info(f"총 {total_tasks}개의 훈련 그룹이 큐(Queue)에 등록되었습니다.")
    logger.info(f"그룹당 시드(모델) 수: {args.count_per_task}개 | 동시 실행(코어) 수: {args.jobs}개")
    for lev, prof in tasks:
        logger.info(f"  - 레버리지: {lev}x | 프로파일: {prof}")
    logger.info(f"{'='*65}\n")
    
    start_time = time.time()
    completed_tasks = 0

    while tasks or active_procs:
        # 슬롯이 비어있으면 큐에서 작업을 꺼내어 새 프로세스 투입
        while tasks and len(active_procs) < args.jobs:
            lev, prof = tasks.popleft()
            
            cmd = [
                sys.executable,
                TRAIN_SCRIPT,
                "--leverage", str(lev),
                "--tuning-profile", prof,
                "--count", str(args.count_per_task),
                "--patience", str(args.patience),
                "--data-path", args.data_path,
                "--model-dir", args.model_dir,
                "--log-dir", args.log_dir
            ]
            
            if args.load_model:
                cmd.extend(["--load-model", args.load_model])
            cmd.extend(["--n-envs", str(args.n_envs)])
            if args.multi_symbol:
                cmd.append("--multi-symbol")
                
            logger.info(f"▶️ [START] 레버리지: {lev}x | 프로파일: {prof:<10} | 시드 투입: {args.count_per_task}개")
            
            # 환경변수 상속 및 프로세스 실행
            # 소형 MLP [256,256,128]: 멀티스레드 BLAS는 동기화 오버헤드 > 병렬 이득
            # 벤치마크: 1 thread=676 iter/s, 20 threads=307 iter/s (2.2x 느림)
            env_vars = os.environ.copy()
            env_vars["PYTHONIOENCODING"] = "utf-8"
            env_vars["OMP_NUM_THREADS"]  = "1"
            env_vars["MKL_NUM_THREADS"]  = "1"
            
            proc = subprocess.Popen(cmd, 
                env=env_vars, 
                cwd=ROOT_DIR,
                # stdout=subprocess.DEVNULL, # 👈 로그 출력 숨김
                stderr=subprocess.DEVNULL  # 👈 에러 출력 숨김
            )
            active_procs.append((proc, lev, prof))
        
        # 프로세스 상태 모니터링
        for proc, lev, prof in active_procs[:]:
            ret = proc.poll()
            
            if ret is not None:
                active_procs.remove((proc, lev, prof))
                completed_tasks += 1
                
                # 에러 발생 시 즉시 중단 안전장치
                if ret != 0:
                    logger.error(f"\n[ERROR] ❌ {lev}x ({prof}) 훈련 그룹에서 치명적 에러 발생! (Exit code: {ret})")
                    for p, _, _ in active_procs:
                        p.terminate()
                    sys.exit(ret)
                else:
                    logger.info(f"✅ [DONE] 레버리지: {lev}x | 프로파일: {prof:<10} (완료: {completed_tasks}/{total_tasks})")
        
        time.sleep(1) 

    total_elapsed = time.time() - start_time
    hours, rem = divmod(total_elapsed, 3600)
    mins, secs = divmod(rem, 60)
    
    logger.info("")
    logger.info(f"{'='*65}")
    logger.info(f"🎉 훈련 배치가 모두 종료되었습니다!")
    logger.info(f"⏱️ 배치 단위 총 소요 시간: {int(hours)}시간 {int(mins)}분 {int(secs)}초")
    logger.info(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="병렬 RL 훈련 오케스트레이터 (백테스트 분리형)")
    
    parser.add_argument("--leverages", type=str, default="1,3,5", help="쉼표로 구분된 레버리지 목록 (예: 1,3,5)")
    parser.add_argument("--profiles", type=str, default="stable,balanced,aggressive", help="쉼표로 구분된 튜닝 프로파일")
    parser.add_argument("--count-per-task", type=int, default=10, help="각 그룹(조합)당 훈련할 모델 수 (기본: 10)")
    parser.add_argument("--jobs", type=int, default=get_optimal_jobs(), help="동시 실행할 병렬 프로세스 수 (기본: CPU코어//2 자동 감지)")
    parser.add_argument("--patience", type=int, default=50, help="03_train_rl.py로 전달할 조기종료 인내심 값")
    
    parser.add_argument("--data-path", type=str, default=os.path.join("data", "signals", "base_signals_log.csv"))
    parser.add_argument("--model-dir", type=str, default=os.path.join("checkpoints", "rl_generations"))
    parser.add_argument("--log-dir", type=str, default=os.path.join("logs", "train"))
    parser.add_argument("--load-model", type=str, default=None, help="(파인튜닝용) 부모 모델 zip 경로")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="DummyVecEnv 병렬 환경 수 (기본=4). CPU 소형 MLP 최적값.")
    parser.add_argument(
        "--multi-symbol", action="store_true", default=False,
        help="BTC/ETH/SOL/XRP 4개 심볼 신호를 모두 사용해 학습합니다 (데이터 다양성 확보).\n"
             "활성화 시 --data-path는 eval 환경(BTC 기준)에만 사용됩니다."
    )

    args = parser.parse_args()

    # 환경변수가 있으면 우선 적용 (run_evolution.py 호환성)
    args.model_dir  = os.environ.get("CUSTOM_MODEL_DIR",  args.model_dir)
    args.log_dir    = os.environ.get("CUSTOM_LOG_DIR",    args.log_dir)
    args.data_path  = os.environ.get("CUSTOM_DATA_PATH",  args.data_path)
    if os.environ.get("MULTI_SYMBOL", "").lower() in ("1", "true", "yes"):
        args.multi_symbol = True
    
    run_parallel_orchestrator(args)