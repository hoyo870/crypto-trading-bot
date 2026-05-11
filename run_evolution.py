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
import time
import argparse
import subprocess
import shutil
import pandas as pd
import glob
from pathlib import Path
import logging

# ── 경로 설정 ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
TRAIN_BATCH_SCRIPT = os.path.join(SCRIPTS_DIR, "04_train_rl_batch.py")
BACKTEST_SCRIPT = os.path.join(SCRIPTS_DIR, "07_backtest_batch.py")

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
logger = logging.getLogger("Evolution")


def _safe_tag_fragment(text: str) -> str:
    """파일명/태그에서 안전하게 쓸 수 있는 조각으로 변환합니다."""
    raw = str(text or "").strip().lower()
    cleaned = [c if c.isalnum() or c in ("_", "-") else "_" for c in raw]
    return "".join(cleaned).strip("_") or "parent"


def _inject_parent_candidate(parent_path: str, gen_str: str, gen_model_dir: str) -> str | None:
    """부모 모델을 현재 세대 비교군으로 추가하고 tags.txt 에 등록합니다."""
    if not parent_path:
        return None
    if not os.path.isfile(parent_path):
        logger.warning(f"⚠️ [경고] --initial-parent 경로를 찾을 수 없어 비교군 추가를 건너뜁니다: {parent_path}")
        return None

    stem = _safe_tag_fragment(Path(parent_path).stem)
    parent_tag = f"parent_{gen_str}_{stem}"
    parent_zip_path = os.path.join(gen_model_dir, f"{parent_tag}.zip")

    if os.path.abspath(parent_path) != os.path.abspath(parent_zip_path):
        shutil.copy2(parent_path, parent_zip_path)
        logger.info(f"🧬 [{gen_str}] 부모 비교군 모델 복사: {parent_zip_path}")
    else:
        logger.info(f"🧬 [{gen_str}] 부모 비교군 모델 재사용: {parent_zip_path}")

    tags_path = os.path.join(gen_model_dir, "tags.txt")
    existing_tags = []
    if os.path.exists(tags_path):
        with open(tags_path, "r", encoding="utf-8") as f:
            existing_tags = [line.strip() for line in f if line.strip()]

    if parent_tag not in existing_tags:
        existing_tags.append(parent_tag)
        with open(tags_path, "w", encoding="utf-8") as f:
            f.write("\n".join(existing_tags) + "\n")
        logger.info(f"🏷️ [{gen_str}] tags.txt 부모 비교군 추가: {parent_tag}")

    return parent_tag


def _find_row_by_tag(reports_dir: str, tag: str):
    """best_by_leverage.csv 에서 특정 tag 행을 찾아 반환합니다."""
    csv_path = os.path.join(reports_dir, "best_by_leverage.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "tag" not in df.columns:
        return None
    rows = df[df["tag"] == tag]
    if rows.empty:
        return None
    return rows.iloc[0]


def _should_promote_over_parent(
    reports_dir: str,
    winner_tag: str,
    parent_tag: str,
    min_score_margin: float,
    max_mdd_delta: float,
):
    """
    부모 대비 비퇴보 게이트.

    승격 조건:
      1) winner_score >= parent_score + min_score_margin
      2) abs(winner_mdd) <= abs(parent_mdd) + max_mdd_delta
      3) winner_sharpe >= parent_sharpe
    """
    winner_row = _find_row_by_tag(reports_dir, winner_tag)
    parent_row = _find_row_by_tag(reports_dir, parent_tag)

    if winner_row is None:
        return False, f"우승 후보 행을 찾지 못함(tag={winner_tag})"
    if parent_row is None:
        return False, f"부모 비교군 행을 찾지 못함(tag={parent_tag})"

    winner_score = float(winner_row.get("metric_value", float("-inf")))
    parent_score = float(parent_row.get("metric_value", float("-inf")))
    winner_mdd = abs(float(winner_row.get("mdd_pct", 999.0)))
    parent_mdd = abs(float(parent_row.get("mdd_pct", 999.0)))
    winner_sharpe = float(winner_row.get("sharpe_ratio", float("-inf")))
    parent_sharpe = float(parent_row.get("sharpe_ratio", float("-inf")))

    cond_score = winner_score >= (parent_score + min_score_margin)
    cond_mdd = winner_mdd <= (parent_mdd + max_mdd_delta)
    cond_sharpe = winner_sharpe >= parent_sharpe

    if cond_score and cond_mdd and cond_sharpe:
        return True, (
            f"승격 통과(score {winner_score:.3f} >= {parent_score + min_score_margin:.3f}, "
            f"MDD {winner_mdd:.2f} <= {parent_mdd + max_mdd_delta:.2f}, "
            f"Sharpe {winner_sharpe:.3f} >= {parent_sharpe:.3f})"
        )

    reasons = []
    if not cond_score:
        reasons.append(f"score 미달({winner_score:.3f} < {parent_score + min_score_margin:.3f})")
    if not cond_mdd:
        reasons.append(f"MDD 악화({winner_mdd:.2f} > {parent_mdd + max_mdd_delta:.2f})")
    if not cond_sharpe:
        reasons.append(f"Sharpe 미달({winner_sharpe:.3f} < {parent_sharpe:.3f})")
    return False, ", ".join(reasons)

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
def auto_discard_models(reports_dir, model_dir, log_dir, keep_top_k):
    """백테스트 결과를 읽어 Top K에 들지 못한 모델과 로그, 차트를 물리적으로 완벽히 삭제합니다."""
    summary_file = os.path.join(reports_dir, "best_by_leverage.csv")
    if not os.path.exists(summary_file):
        logger.warning("⚠️ [경고] 백테스트 요약 파일이 없어 폐기 작업을 건너뜁니다.")
        return None

    df = pd.read_csv(summary_file)
    df = df.sort_values(by="metric_value", ascending=False).reset_index(drop=True)
    survivors = df.head(keep_top_k)["tag"].tolist()
    best_model_tag = survivors[0] if survivors else None
    
    logger.info("")
    logger.info(f"🧹 [자동 폐기 가동] 생존자 탑 {keep_top_k}명 선정 (총 {len(survivors)}개)")
    if len(survivors) <= 10:
        logger.info(f"   생존자: {survivors}")
    else:
        logger.info(f"   생존자 (처음 10개): {survivors[:10]} ... 외 {len(survivors)-10}개")
    
    deleted_count = 0
    for item in os.listdir(model_dir):
        if item.startswith("."): continue
        item_tag = item[:-4] if item.endswith(".zip") else item
        
        # 💀 생존자 명단에 없으면 무자비하게 3단 삭제 (가중치/리포트/로그)
        if item_tag not in survivors and item_tag != "best_by_leverage.csv" and not item_tag.startswith("best_gen"):
            # 1. 모델 가중치 삭제
            target_path = os.path.join(model_dir, item)
            if os.path.isdir(target_path): shutil.rmtree(target_path)
            else: os.remove(target_path)
            
            # 2. 리포트 및 차트 파일 싹쓸이 (Python glob 사용으로 안정성 100%)
            for file_path in glob.glob(os.path.join(reports_dir, f"*{item_tag}*")):
                try: os.remove(file_path)
                except Exception: pass
            
            # 3. 텐서보드 로그 폴더 싹쓸이
            for log_folder in glob.glob(os.path.join(log_dir, f"{item_tag}*")):
                try: shutil.rmtree(log_folder)
                except Exception: pass
                
            deleted_count += 1
            
    logger.info(f"✅ 총 {deleted_count}개의 열등한 모델(가중치/차트/로그)이 완벽히 소각되었습니다.\n")
    return best_model_tag


# ── 3. 메인 오케스트레이터 루프 ──────────────────────────────────────────────
def run_evolution_pipeline(args):
    pipeline_start_time = time.time()
    
    checkpoints_root = os.path.join(ROOT_DIR, "checkpoints", "rl_generations")
    reports_root = os.path.join(ROOT_DIR, "reports")
    logs_root = os.path.join(ROOT_DIR, "logs", "train")
    
    current_gen = get_current_generation(checkpoints_root)
    start_gen = current_gen
    end_gen = start_gen + args.target_generations - 1

    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"🧬 [마법의 진화 파이프라인 가동] 목표: {start_gen}세대 ➡️ {end_gen}세대")
    logger.info(f"{'='*70}")

    parent_model_path = args.initial_parent  # 1세대 시작 시 외부 모델을 꽂고 싶을 때
    gen_reports_dir = reports_root            # 루프 실행 전 기본값 (NameError 방지)

    for gen in range(start_gen, end_gen + 1):
        gen_start_time = time.time()
        incoming_parent_path = parent_model_path
        parent_candidate_tag = None
        
        gen_str = f"gen{gen}"
        gen_model_dir = os.path.join(checkpoints_root, gen_str)
        gen_reports_dir = os.path.join(reports_root, gen_str)
        gen_logs_dir = os.path.join(logs_root, gen_str)
        os.makedirs(gen_model_dir, exist_ok=True)
        os.makedirs(gen_reports_dir, exist_ok=True)
        os.makedirs(gen_logs_dir, exist_ok=True)
        
        logger.info("")
        logger.info(f"🌱 [시작] {gen_str} 세대 배양을 시작합니다...")
        
        # --- [Phase 1: 병렬 훈련 가동] ---
        train_cmd = [
            sys.executable, TRAIN_BATCH_SCRIPT,
            "--count-per-task", str(args.count_per_task),
            "--jobs", str(args.jobs),
            "--leverages", args.leverages,
            "--profiles", args.profiles
        ]
        if parent_model_path:
            train_cmd.extend(["--load-model", parent_model_path])
            
        # 04_train_rl_batch.py에 환경 변수로 현재 세대 경로를 넘겨줌
        env_vars = os.environ.copy()
        env_vars["CUSTOM_MODEL_DIR"] = gen_model_dir
        env_vars["CUSTOM_LOG_DIR"] = gen_logs_dir
        env_vars["PYTHONIOENCODING"] = "utf-8"
        
        # 04_train 스크립트 실행 (백테스트는 파이프라인에서 직접 통제하므로 no-backtest 옵션 추가 요망)
        result = subprocess.run(train_cmd, env=env_vars, cwd=ROOT_DIR)
        if result.returncode != 0:
            logger.error(f"❌ [{gen_str}] 훈련 스크립트 실패 (exit {result.returncode}). 진화를 중단합니다.")
            break

        # 초기/이전 세대 부모를 현재 세대 백테스트 비교군에 포함
        parent_candidate_tag = _inject_parent_candidate(incoming_parent_path, gen_str, gen_model_dir)
        
        # --- [Phase 2: 백테스트 및 평가] ---
        logger.info("")
        logger.info(f"📊 [{gen_str}] 훈련 완료. 전체 백테스트 및 성적표 산출 중...")
        bt_cmd = [
            sys.executable, BACKTEST_SCRIPT,
            "--model-dir",   gen_model_dir,
            "--reports-dir", gen_reports_dir,
            "--jobs",        str(args.jobs),
            "--metric",      args.metric,
            "--formula",     args.formula,
        ]
        # tags.txt 가 model_dir 에 있으면 07_backtest_batch.py 가 자동으로 읽음
        result = subprocess.run(bt_cmd, env=env_vars, cwd=ROOT_DIR)
        if result.returncode != 0:
            logger.error(f"❌ [{gen_str}] 백테스트 스크립트 실패 (exit {result.returncode}). 진화를 중단합니다.")
            break
        
        # --- [Phase 3: 자동 폐기 및 다음 세대 부모 선발] ---
        best_tag = auto_discard_models(gen_reports_dir, gen_model_dir, gen_logs_dir, args.auto_discard_top)
        
        if best_tag:
            # 1. 1등 모델의 zip 파일 경로를 정확히 탐색
            winner_source_path = os.path.join(gen_model_dir, f"{best_tag}.zip")
            if not os.path.exists(winner_source_path):
                # 폴더 안에 저장된 경우의 Fallback
                winner_source_path = os.path.join(gen_model_dir, best_tag, f"final_model_{best_tag}.zip")
                
            logger.info(f"👑 [{gen_str}] 최종 우승자: {best_tag}")

            # 부모 모델이 그대로 우승한 경우는 best_gen 복사를 생략
            if parent_candidate_tag and best_tag == parent_candidate_tag:
                parent_model_path = incoming_parent_path
                logger.info(f"⏭️ [{gen_str}] 부모 모델이 계속 우승하여 best_{gen_str}.zip 복사를 생략합니다.")
                logger.info(f"➡️ 동일 부모 모델을 {gen+1}세대 유전자로 유지합니다.")
                gen_elapsed = time.time() - gen_start_time
                logger.info(f"✨ [{gen_str}] 세대 완료! (소요 시간: {int(gen_elapsed//60)}분)")
                continue

            # 부모 비교군이 있으면 비퇴보 게이트 통과 시에만 승격
            if parent_candidate_tag and not args.disable_parent_gate:
                promote_ok, gate_reason = _should_promote_over_parent(
                    reports_dir=gen_reports_dir,
                    winner_tag=best_tag,
                    parent_tag=parent_candidate_tag,
                    min_score_margin=args.parent_score_margin,
                    max_mdd_delta=args.parent_max_mdd_delta,
                )
                if not promote_ok:
                    parent_model_path = incoming_parent_path
                    logger.info(f"🛡️ [{gen_str}] 부모 비퇴보 게이트 미통과: {gate_reason}")
                    logger.info(f"⏭️ [{gen_str}] 우승자 갱신/복사를 생략하고 기존 부모를 유지합니다.")
                    gen_elapsed = time.time() - gen_start_time
                    logger.info(f"✨ [{gen_str}] 세대 완료! (소요 시간: {int(gen_elapsed//60)}분)")
                    continue
                logger.info(f"✅ [{gen_str}] 부모 비퇴보 게이트 통과: {gate_reason}")

            # ✅ 2. 우승 모델 복사 로직
            winner_dst = os.path.join(gen_model_dir, f"best_{gen_str}.zip")
            if os.path.exists(winner_source_path):
                if os.path.abspath(winner_source_path) == os.path.abspath(winner_dst):
                    logger.info(f"ℹ️ [{gen_str}] 우승 모델 경로와 목적지가 동일하여 복사를 건너뜁니다.")
                else:
                    shutil.copy2(winner_source_path, winner_dst)
                logger.info(f"📂 [복사완료] 우승 모델이 {winner_dst} 로 복사(보존) 되었습니다.")
                
                # 다음 세대로 넘겨줄 경로를, 방금 예쁘게 복사한 best_gen.zip 으로 교체!
                parent_model_path = winner_dst
            else:
                logger.error(f"❌ [{gen_str}] 우승 모델 파일을 찾지 못했습니다: {winner_source_path}")
                break
            logger.info(f"➡️ 이 모델이 {gen+1}세대의 부모 유전자로 투입됩니다.")
        else:
            logger.error("❌ [치명적 에러] 생존한 모델이 없습니다. 진화를 중단합니다.")
            break
            
        gen_elapsed = time.time() - gen_start_time
        logger.info(f"✨ [{gen_str}] 세대 완료! (소요 시간: {int(gen_elapsed//60)}분)")

    total_elapsed = time.time() - pipeline_start_time
    hours, rem = divmod(total_elapsed, 3600)
    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"🎉 모든 진화 과정이 완료되었습니다! (총 소요 시간: {int(hours)}시간 {int(rem//60)}분)")
    logger.info(f"최종 우승 모델을 {gen_reports_dir} 에서 확인하세요.")
    logger.info(f"{'='*70}")

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
#    ▶ python run_evolution.py --target-generations 3 --auto-discard-top 1 --leverages 1,3,5 --profiles stable,balanced,aggressive --count-per-task 33
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
  python run_evolution.py --target-generations 1 --count-per-task 3 --auto-discard-top 999 --leverages 1,1,1 --profiles balanced --initial-parent "checkpoints/rl_generations/gen1/best_gen1.zip"
  python run_evolution.py --target-generations 3 --auto-discard-top 1 --leverages 1,3,5 --profiles stable,balanced,aggressive --count-per-task 33
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
    parser.add_argument("--jobs", type=int, default=get_optimal_jobs(), 
                        help="동시 실행할 병렬 프로세스 수 (기본: CPU코어//2 자동 감지)")
    parser.add_argument("--leverages", type=str, default="1,3,5", 
                        help="훈련할 레버리지 목록 (쉼표 구분, 기본: 1,3,5)")
    parser.add_argument("--profiles", type=str, default="stable,balanced,aggressive", 
                        help="훈련할 프로파일 목록 (쉼표 구분, 기본: stable,balanced,aggressive)")
    
    # 자동 폐기 옵션
    parser.add_argument("--auto-discard-top", type=int, default=3, 
                        help="백테스트 순위 1등 ~ K등까지만 살리고 나머지 폴더/가중치/로그 삭제 (기본: 3)\n"
                             "(전부 살리려면 999 같은 큰 숫자를 입력하세요)")
    
    # 랭킹 옵션
    parser.add_argument("--metric",
                        choices=["score", "total_return_pct", "sharpe_ratio", "mdd_pct"],
                        default="score",
                        help="랭킹 기준 지표 (기본: score)")
    parser.add_argument("--formula",
                        choices=["balanced", "aggressive", "conservative"],
                        default="balanced",
                        help="점수 계산 공식 (--metric score일 때만 사용, 기본: balanced)")
    parser.add_argument("--disable-parent-gate", action="store_true",
                        help="부모 비교군 비퇴보 게이트 비활성화 (기본: 활성)")
    parser.add_argument("--parent-score-margin", type=float, default=1.0,
                        help="부모 대비 최소 점수 향상 폭 (기본: 1.0)")
    parser.add_argument("--parent-max-mdd-delta", type=float, default=2.0,
                        help="부모 대비 허용 가능한 MDD 악화 폭 %%p (기본: 2.0)")
    
    args = parser.parse_args()
    
    # 안전장치: 폐기 수를 1 미만으로 적었을 때의 버그 방지
    if args.auto_discard_top < 1:
        logger.warning("⚠️ [경고] --auto-discard-top 값은 최소 1 이상이어야 합니다. 강제로 1로 조정합니다.")
        args.auto_discard_top = 1
        
    run_evolution_pipeline(args)