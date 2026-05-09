"""
마법의 원클릭 진화 파이프라인 (Auto MLOps Orchestrator)

사용 예시:
  # 1. 기본 1세대만 가동하고 성적 안 좋은 모델은 다 폐기 (Top 3만 생존)
  python run_evolution.py --auto-discard-top 3

  # 2. 1세대부터 3세대까지 자동 진화 (1등 모델을 다음 세대 부모로 자동 투입)
  python run_evolution.py --target-generations 3 --auto-discard-top 1
"""

import os
import sys
import argparse
import subprocess
import shutil
import pandas as pd
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
TRAIN_BATCH_SCRIPT = os.path.join(SCRIPTS_DIR, "04_train_rl_batch.py")
BACKTEST_SCRIPT = os.path.join(SCRIPTS_DIR, "05_backtest.py")

# ── 1. 세대(Generation) 스캐너 ─────────────────────────────────────────────
def get_current_generation(checkpoints_dir):
    """현재 폴더 상태를 읽어 몇 세대까지 진행되었는지 파악합니다."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    gens = [d for d in os.listdir(checkpoints_dir) if d.startswith("gen") and os.path.isdir(os.path.join(checkpoints_dir, d))]
    if not gens:
        return 1
    # gen1, gen2 중 가장 높은 숫자 반환
    return max([int(g.replace("gen", "")) for g in gens])

# ── 2. 자동 폐기(Garbage Collector) ─────────────────────────────────────────
def auto_discard_models(reports_dir, model_dir, keep_top_k):
    """백테스트 결과를 읽어 Top K에 들지 못한 모델과 로그, 차트를 물리적으로 삭제합니다."""
    summary_file = os.path.join(reports_dir, "best_by_leverage.csv")
    if not os.path.exists(summary_file):
        print("⚠️ [경고] 백테스트 요약 파일을 찾을 수 없어 폐기 작업을 건너뜁니다.")
        return None

    # CSV 읽기 (metric_value가 높은 순으로 정렬)
    df = pd.read_csv(summary_file)
    df = df.sort_values(by="metric_value", ascending=False).reset_index(drop=True)
    
    survivors = df.head(keep_top_k)["tag"].tolist()
    best_model_tag = survivors[0] if survivors else None
    
    print(f"\n🧹 [자동 폐기 가동] 생존자 탑 {keep_top_k}명: {survivors}")
    
    deleted_count = 0
    # 모델 폴더 스캔 및 폐기
    for item in os.listdir(model_dir):
        if item.startswith("."): continue
        item_tag = item[:-4] if item.endswith(".zip") else item
        
        # 생존자 명단에 없으면 무자비하게 삭제
        if item_tag not in survivors and item_tag != "best_by_leverage.csv":
            target_path = os.path.join(model_dir, item)
            
            # 모델 가중치(zip) 및 폴더 삭제
            if os.path.isdir(target_path): shutil.rmtree(target_path)
            else: os.remove(target_path)
            
            # 찌꺼기 리포트/차트도 삭제
            for ext in [".json", ".txt", ".png"]:
                report_trash = os.path.join(reports_dir, f"rl_backtest_*{item_tag}*{ext}")
                os.system(f"rm -f {report_trash}")
                
            deleted_count += 1
            
    print(f"✅ 총 {deleted_count}개의 열등한 모델과 찌꺼기 리포트가 소각되었습니다.\n")
    return best_model_tag


# ── 3. 메인 오케스트레이터 루프 ──────────────────────────────────────────────
def run_evolution_pipeline(args):
    checkpoints_root = os.path.join(ROOT_DIR, "checkpoints", "rl_generations")
    reports_root = os.path.join(ROOT_DIR, "reports")
    
    current_gen = get_current_generation(checkpoints_root)
    start_gen = current_gen
    end_gen = start_gen + args.target_generations - 1

    print(f"\n{'='*70}")
    print(f"🧬 [마법의 진화 파이프라인 가동] 목표: {start_gen}세대 ➡️ {end_gen}세대")
    print(f"{'='*70}")

    parent_model_path = args.initial_parent # 1세대 시작 시 외부 모델을 꽂고 싶을 때

    for gen in range(start_gen, end_gen + 1):
        gen_str = f"gen{gen}"
        gen_model_dir = os.path.join(checkpoints_root, gen_str)
        gen_reports_dir = os.path.join(reports_root, gen_str)
        
        os.makedirs(gen_model_dir, exist_ok=True)
        os.makedirs(gen_reports_dir, exist_ok=True)
        
        print(f"\n🌱 [시작] {gen_str} 세대 배양을 시작합니다...")
        
        # --- [Phase 1: 병렬 훈련 가동] ---
        train_cmd = [
            sys.executable, TRAIN_BATCH_SCRIPT,
            "--count-per-task", str(args.count_per_task),
            "--jobs", str(args.jobs)
        ]
        if parent_model_path:
            train_cmd.extend(["--load-model", parent_model_path])
            
        # 04_train_rl_batch.py에 환경 변수로 현재 세대 경로를 넘겨줌
        env_vars = os.environ.copy()
        env_vars["CUSTOM_MODEL_DIR"] = gen_model_dir
        
        # 04_train 스크립트 실행 (백테스트는 파이프라인에서 직접 통제하므로 no-backtest 옵션 추가 요망)
        subprocess.run(train_cmd, env=env_vars, cwd=ROOT_DIR)
        
        # --- [Phase 2: 백테스트 및 평가] ---
        print(f"\n📊 [{gen_str}] 훈련 완료. 전체 백테스트 및 성적표 산출 중...")
        bt_cmd = [
            sys.executable, BACKTEST_SCRIPT,
            "--model-dir", gen_model_dir,
            "--reports-dir", gen_reports_dir
        ]
        subprocess.run(bt_cmd, cwd=ROOT_DIR)
        
        # --- [Phase 3: 자동 폐기 및 다음 세대 부모 선발] ---
        best_tag = auto_discard_models(gen_reports_dir, gen_model_dir, args.auto_discard_top)
        
        if best_tag:
            # 1등 모델의 zip 파일 경로를 다음 세대의 부모로 지정
            parent_model_path = os.path.join(gen_model_dir, f"{best_tag}.zip")
            if not os.path.exists(parent_model_path):
                # 폴더 안에 저장된 경우
                parent_model_path = os.path.join(gen_model_dir, best_tag, f"final_model_{best_tag}.zip")
                
            print(f"👑 [{gen_str}] 최종 우승자: {best_tag}")
            print(f"➡️ 이 모델이 {gen+1}세대의 부모 유전자로 투입됩니다.")
        else:
            print("❌ [치명적 에러] 생존한 모델이 없습니다. 진화를 중단합니다.")
            break

    print(f"\n{'='*70}")
    print(f"🎉 모든 진화 과정이 완료되었습니다! 최종 우승 모델을 {gen_reports_dir} 에서 확인하세요.")
    print(f"{'='*70}")

# ─────────────────────────────────────────────────────────────────────────────
# 💡 [마법의 진화 파이프라인 사용법 (Scenarios)]
# ─────────────────────────────────────────────────────────────────────────────
#
# 1️⃣ 시나리오 1: "일단 1세대만 가볍게 돌려보자" (단일 세대 배양)
#    ▶ python run_evolution.py --target-generations 1 --count-per-task 10
#    - 설명: 현재 세대(Gen1)만 훈련합니다. 각 조합당 10개씩 훈련하고, 
#            백테스트 결과 1~3등(기본값)만 남기고 나머지는 자동 폐기합니다.
#
# 2️⃣ 시나리오 2: "내일 아침까지 알아서 3세대 진화시켜놔" (연속 세대 진화)
#    ▶ python run_evolution.py --target-generations 3 --auto-discard-top 1 --count-per-task 33
#    - 설명: Gen1 훈련 -> 백테스트 1등 선발 (나머지 삭제) -> 1등 뇌를 Gen2에 이식 ->
#            Gen2 훈련 -> 백테스트 1등 선발 (나머지 삭제) -> 1등 뇌를 Gen3에 이식.
#            내일 아침에는 Gen3의 궁극체 1개만 폴더에 남게 됩니다.
#
# 3️⃣ 시나리오 3: "과거의 전설적인 모델을 데려와서 거기서부터 진화시키고 싶어"
#    ▶ python run_evolution.py --initial-parent "checkpoints/legacy/legend.zip" --target-generations 2
#    - 설명: 무작위 가중치(백지)가 아니라, 지정된 부모 모델의 뇌를 이식받은 상태로 
#            새로운 환경에서 진화를 시작합니다. (전이 학습/파인튜닝)
#
# 4️⃣ 시나리오 4: "데이터 폐기 없이 전부 다 살려둬!" (자동 폐기 끄기)
#    ▶ python run_evolution.py --auto-discard-top 999
#    - 설명: 폐기 기준을 엄청 높게 잡아서, 생성된 모든 가중치와 리포트를 지우지 않고 보존합니다.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🧬 Commander Auto MLOps Orchestrator (원클릭 진화 파이프라인)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
[실행 예시]
  python run_evolution.py --target-generations 1 --count-per-task 10
  python run_evolution.py --target-generations 3 --auto-discard-top 1
  python run_evolution.py --initial-parent "checkpoints/rl_generations/gen1/best.zip"
        """
    )
    
    # 세대 진화 옵션
    parser.add_argument("--target-generations", type=int, default=1, 
                        help="이번 실행에서 전진시킬 세대 수 (기본: 1)\n"
                             "예: 현재가 gen1일 때 3을 입력하면 gen3까지 연속 진화함")
    parser.add_argument("--initial-parent", type=str, default=None, 
                        help="최초 1세대 훈련 시 물려줄 뇌(가중치 .zip)가 있다면 경로 지정\n"
                             "(미지정 시 무작위 가중치로 백지에서 시작)")
    
    # 훈련 큐 옵션
    parser.add_argument("--count-per-task", type=int, default=10, 
                        help="각 (레버리지, 프로파일) 조합당 훈련할 씨앗(Seed) 개체 수 (기본: 10)")
    parser.add_argument("--jobs", type=int, default=3, 
                        help="동시 실행할 병렬 프로세스 수 (M1 Max 권장: 3)")
    
    # 자동 폐기 옵션
    parser.add_argument("--auto-discard-top", type=int, default=3, 
                        help="백테스트 순위 1등 ~ K등까지만 살리고 나머지 폴더/가중치/로그 삭제 (기본: 3)\n"
                             "(전부 살리려면 999 같은 큰 숫자를 입력하세요)")
    
    args = parser.parse_args()
    
    # 안전장치: 폐기 수를 1 미만으로 적었을 때의 버그 방지
    if args.auto_discard_top < 1:
        print("⚠️ [경고] --auto-discard-top 값은 최소 1 이상이어야 합니다. 강제로 1로 조정합니다.")
        args.auto_discard_top = 1
        
    run_evolution_pipeline(args)