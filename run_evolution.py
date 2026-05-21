"""
마법의 원클릭 진화 파이프라인 (Auto MLOps Orchestrator)

사용 예시:
  # 1. 기본 1세대만 가동하고 성적 안 좋은 모델은 다 폐기 (Top 3만 생존)
  python run_evolution.py --auto-discard-top 3

  # 2. 1세대부터 3세대까지 자동 진화 (1등 모델을 다음 세대 부모로 자동 투입)
  python run_evolution.py --target-generations 3 --auto-discard-top 1
"""

import os
import re
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
BACKTEST_SCRIPT    = os.path.join(SCRIPTS_DIR, "07_backtest_batch.py")
FINETUNE_SCRIPT    = os.path.join(SCRIPTS_DIR, "03b_finetune_full.py")

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


def _extract_original_tag(parent_path: str) -> str | None:
    """
    best_genX_lev2_bal_seed12345_001.zip 처럼 best_genX_ 접두사가 붙은 경로에서
    원본 태그(lev2_bal_seed12345_001)를 추출합니다.
    원본 태그가 없으면 None 반환.
    """
    stem = Path(parent_path).stem  # 확장자 제거
    # best_genX_ 패턴 제거
    m = re.match(r"^best_gen\d+_(.+)$", stem)
    if m:
        return m.group(1)
    # lev로 시작하는 경우 그대로 사용
    if re.match(r"^lev\d+_", stem):
        return stem
    return None


def _inject_parent_candidate(parent_path: str, gen_str: str, gen_model_dir: str) -> str | None:
    """부모 모델을 현재 세대 비교군으로 추가하고 tags.txt 에 등록합니다.

    원본 태그(lev{N}_{prof}_...) 정보를 보존하여 07_backtest_batch.py 가
    정확한 레버리지와 프로파일을 추론할 수 있게 합니다.
    """
    if not parent_path:
        return None
    if not os.path.isfile(parent_path):
        logger.warning(f"⚠️ [경고] --initial-parent 경로를 찾을 수 없어 비교군 추가를 건너뜁니다: {parent_path}")
        return None

    # 원본 태그 추출 시도 (best_genX_lev... 패턴에서 lev... 부분만)
    original_tag = _extract_original_tag(parent_path)
    if original_tag:
        # lev{N}_ 접두사가 살아있는 태그: parent_{gen_str}_lev{N}_{prof}_...
        stem = _safe_tag_fragment(original_tag)
    else:
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

# ── 1b. Phase 1.5 — Baby→Full 커리큘럼 파인튜닝 ───────────────────────────
def _run_finetune_phase(gen_model_dir, gen_logs_dir, gen_str, args, env_vars):
    """
    tags.txt 의 Baby 모델 전체를 03b_finetune_full.py 로 Full 환경에 파인튜닝.
    파인튜닝된 모델은 gen_model_dir 안에 저장되고 tags.txt 에 태그가 추가되어
    Phase 2 백테스트 풀에 자동 포함된다.
    """
    from src.utils.model_utils import resolve_model_path

    tags_path = os.path.join(gen_model_dir, "tags.txt")
    if not os.path.exists(tags_path):
        logger.warning(f"⚠️ [{gen_str}] tags.txt 없음 → Phase 1.5 파인튜닝 건너뜀")
        return

    with open(tags_path, "r", encoding="utf-8") as f:
        baby_tags = [line.strip() for line in f if line.strip()]

    if not baby_tags:
        logger.warning(f"⚠️ [{gen_str}] tags.txt 가 비어 있음 → Phase 1.5 파인튜닝 건너뜀")
        return

    # 데이터 경로 결정 (인자 우선, 없으면 환경변수)
    data_path = args.data_path or env_vars.get("CUSTOM_DATA_PATH")
    if not data_path:
        logger.warning(f"⚠️ [{gen_str}] --data-path 미지정 → Phase 1.5 파인튜닝 건너뜀")
        return

    finetune_log_dir = os.path.join(gen_logs_dir, "finetune")
    os.makedirs(finetune_log_dir, exist_ok=True)

    logger.info("")
    logger.info(f"🔧 [{gen_str}] Phase 1.5: {len(baby_tags)}개 Baby 모델 → Full 환경 파인튜닝 시작")

    new_tags = []
    for tag in baby_tags:
        baby_path = resolve_model_path(tag, gen_model_dir)
        if not baby_path:
            logger.warning(f"  [{gen_str}] 모델 파일 없음: {tag} → 건너뜀")
            continue

        # 태그에서 레버리지·프로파일 추출 (예: lev2_balanced_seed12345_001)
        m = re.search(r'lev(\d+)_(stable|balanced|aggressive)', tag)
        lev  = int(m.group(1)) if m else 1
        prof = m.group(2) if m else "balanced"

        cmd = [
            sys.executable, FINETUNE_SCRIPT,
            "--baby-model-path", baby_path,
            "--leverage",        str(lev),
            "--tuning-profile",  prof,
            "--data-path",       data_path,
            "--model-dir",       gen_model_dir,
            "--log-dir",         finetune_log_dir,
            "--timesteps",       str(args.finetune_timesteps),
        ]
        logger.info(f"  ▶️ 파인튜닝 시작: {tag} (lev={lev}, prof={prof})")
        result = subprocess.run(cmd, env=env_vars, cwd=ROOT_DIR)
        if result.returncode == 0:
            ft_tag = f"finetune_lev{lev}_{prof}_{Path(baby_path).stem}"
            new_tags.append(ft_tag)
            logger.info(f"  ✅ 파인튜닝 완료: {ft_tag}")
        else:
            logger.warning(f"  ⚠️ 파인튜닝 실패 (exit {result.returncode}): {tag} → 건너뜀")

    # 새 태그를 tags.txt 에 추가
    if new_tags:
        with open(tags_path, "a", encoding="utf-8") as f:
            for t in new_tags:
                f.write(t + "\n")
        logger.info(f"  🏷️  {len(new_tags)}개 파인튜닝 모델이 tags.txt 에 추가 → Phase 2 평가 풀에 합류")
    logger.info(f"🔧 [{gen_str}] Phase 1.5 완료\n")


# ── 2. 자동 폐기(Garbage Collector) ─────────────────────────────────────────
def auto_discard_models(reports_dir, model_dir, log_dir, keep_top_k):
    """백테스트 결과를 읽어 Top K에 들지 못한 모델과 로그, 차트를 물리적으로 완벽히 삭제합니다."""
    summary_file = os.path.join(reports_dir, "best_by_leverage.csv")
    if not os.path.exists(summary_file):
        logger.warning("⚠️ [경고] 백테스트 요약 파일이 없어 폐기 작업을 건너뜁니다.")
        return None, []

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

    # ── tags.txt 동기화: 폐기된 태그 제거 ────────────────────────────────
    tags_path = os.path.join(model_dir, "tags.txt")
    if os.path.exists(tags_path):
        with open(tags_path, "r", encoding="utf-8") as f:
            old_tags = [line.strip() for line in f if line.strip()]
        new_tags = [t for t in old_tags if t in survivors]
        removed_cnt = len(old_tags) - len(new_tags)
        if removed_cnt > 0:
            with open(tags_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_tags) + "\n")
            logger.info(f"🏷️ tags.txt 동기화 완료: {removed_cnt}개 폐기 태그 제거 → {len(new_tags)}개 생존")

    return best_model_tag, survivors


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

        # 적응형 변이 폭: 세대 진행마다 0.1씩 감소, 최소 0.3 보장
        # ⚠️ 03_train_rl.py 의 FINETUNE_SCALE_FACTOR(=0.5) 가 추가로 곱해지므로
        #    파인튜닝 모드의 실제 유효 변이 스케일 = mutation_scale × 0.5 (유효 범위 [0.15, 0.5])
        mutation_scale = max(0.3, args.mutation_scale_start - (gen - start_gen) * 0.1)
        logger.info(f"🔬 [{gen_str}] 변이 폭 스케일: {mutation_scale:.2f} (파인튜닝 시 실효 스케일: {mutation_scale*0.5:.2f})")

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
        if args.data_path:
            train_cmd.extend(["--data-path", args.data_path])
        if args.multi_symbol:
            train_cmd.append("--multi-symbol")

        # 04_train_rl_batch.py에 환경 변수로 현재 세대 경로를 넘겨줌
        env_vars = os.environ.copy()
        env_vars["CUSTOM_MODEL_DIR"] = gen_model_dir
        env_vars["CUSTOM_LOG_DIR"] = gen_logs_dir
        env_vars["MUTATION_SCALE"] = str(mutation_scale)
        env_vars["PYTHONIOENCODING"] = "utf-8"
        if args.data_path:
            env_vars["CUSTOM_DATA_PATH"] = args.data_path
        if args.multi_symbol:
            env_vars["MULTI_SYMBOL"] = "1"
        
        # 04_train 스크립트 실행 (백테스트는 파이프라인에서 직접 통제하므로 no-backtest 옵션 추가 요망)
        result = subprocess.run(train_cmd, env=env_vars, cwd=ROOT_DIR)
        if result.returncode != 0:
            logger.error(f"❌ [{gen_str}] 훈련 스크립트 실패 (exit {result.returncode}). 진화를 중단합니다.")
            break

        # 초기/이전 세대 부모를 현재 세대 백테스트 비교군에 포함
        parent_candidate_tag = _inject_parent_candidate(incoming_parent_path, gen_str, gen_model_dir)

        # --- [Phase 1.5: Baby→Full 커리큘럼 파인튜닝 (선택)] ---
        # tags.txt 의 모든 Baby 모델을 Full 환경으로 파인튜닝 후 평가 풀에 추가
        if not args.skip_finetune:
            _run_finetune_phase(
                gen_model_dir=gen_model_dir,
                gen_logs_dir=gen_logs_dir,
                gen_str=gen_str,
                args=args,
                env_vars=env_vars,
            )

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
        if args.data_path:
            bt_cmd.extend(["--data-path", args.data_path])
        # tags.txt 가 model_dir 에 있으면 07_backtest_batch.py 가 자동으로 읽음
        result = subprocess.run(bt_cmd, env=env_vars, cwd=ROOT_DIR)
        if result.returncode != 0:
            logger.error(f"❌ [{gen_str}] 백테스트 스크립트 실패 (exit {result.returncode}). 진화를 중단합니다.")
            break
        
        # --- [Phase 3: 자동 폐기 및 생존자 선발] ---
        best_tag, survivors = auto_discard_models(gen_reports_dir, gen_model_dir, gen_logs_dir, args.auto_discard_top)

        # --- [Phase 4: 생존 모델 풀 버전 백테스트 (최종 우승자 결정)] ---
        if survivors:
            logger.info("")
            logger.info(f"🏆 [{gen_str}] 생존 모델 {len(survivors)}개 풀({args.final_eval_env}) 백테스트 시작 — 최종 우승자 결정 중...")
            full_bt_cmd = [
                sys.executable, BACKTEST_SCRIPT,
                "--model-dir",   gen_model_dir,
                "--reports-dir", gen_reports_dir,
                "--jobs",        str(args.jobs),
                "--metric",      args.metric,
                "--formula",     args.formula,
                "--tags",        ",".join(survivors),
                "--env-type",    args.final_eval_env,
            ]
            if args.data_path:
                full_bt_cmd.extend(["--data-path", args.data_path])
            full_result = subprocess.run(full_bt_cmd, env=env_vars, cwd=ROOT_DIR)
            if full_result.returncode != 0:
                logger.warning(f"⚠️ [{gen_str}] 풀 백테스트 실패 — 1차 순위({best_tag})를 그대로 사용합니다.")
            else:
                full_csv = os.path.join(gen_reports_dir, "best_by_leverage.csv")
                try:
                    df_full = pd.read_csv(full_csv)
                    df_full = df_full.sort_values("metric_value", ascending=False).reset_index(drop=True)
                    if not df_full.empty and "tag" in df_full.columns:
                        new_best = df_full.iloc[0]["tag"]
                        if new_best != best_tag:
                            logger.info(f"🔄 [{gen_str}] 풀 백테스트 기준 우승자 변경: {best_tag} → {new_best}")
                        else:
                            logger.info(f"✅ [{gen_str}] 풀 백테스트에서도 동일 우승자 유지: {new_best}")
                        best_tag = new_best
                except Exception as e:
                    logger.warning(f"⚠️ [{gen_str}] 풀 백테스트 결과 파싱 실패: {e} — 1차 순위를 그대로 사용합니다.")

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

            # ✅ 2. 우승 모델 복사 로직 — 원본 태그를 파일명에 포함해 보존
            # best_genX_lev{N}_{prof}_{seed}_{idx}.zip 형식으로 저장
            winner_dst = os.path.join(gen_model_dir, f"best_{gen_str}_{best_tag}.zip")
            if os.path.exists(winner_source_path):
                if os.path.abspath(winner_source_path) == os.path.abspath(winner_dst):
                    logger.info(f"ℹ️ [{gen_str}] 우승 모델 경로와 목적지가 동일하여 복사를 건너뜁니다.")
                else:
                    shutil.copy2(winner_source_path, winner_dst)
                logger.info(f"📂 [복사완료] 우승 모델이 {winner_dst} 로 복사(보존) 되었습니다.")
                logger.info(f"🏷️  [태그 보존] 원본 태그 '{best_tag}' 가 파일명에 포함되어 다음 세대에서 레버리지/프로파일이 정확히 추론됩니다.")

                # 다음 세대로 넘겨줄 경로를 원본 태그 포함된 best_genX_TAG.zip 으로 교체
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
  python run_evolution.py --target-generations 3 --auto-discard-top 10 --leverages 1,2 --profiles balanced,aggressive --count-per-task 50
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
    parser.add_argument("--data-path", type=str, default=None,
                        help="RL 훈련/백테스트 데이터 CSV 경로\n"
                             "미지정 시 data/signals/base_signals_log.csv 사용 (BTC_USDT val+test)\n"
                             "다른 코인 예: data/signals/ETH_USDT_signals_log.csv")
    parser.add_argument(
        "--multi-symbol", action="store_true", default=False,
        help="BTC/ETH/SOL/XRP 4개 심볼 신호를 모두 사용해 학습합니다 (데이터 다양성 확보).\n"
             "활성화 시 각 심볼의 {SYM}_signals_log.csv 가 최신 BASE 모델로 재추출돼 있어야 합니다."
    )
    parser.add_argument("--mutation-scale-start", type=float, default=1.0,
                        help="초기 변이 폭 스케일 (1.0=최대, 세대마다 0.1 감소, 최소 0.3)")
    parser.add_argument("--disable-parent-gate", action="store_true",
                        help="부모 비교군 비퇴보 게이트 비활성화 (기본: 활성)")
    parser.add_argument("--parent-score-margin", type=float, default=1.0,
                        help="부모 대비 최소 점수 향상 폭 (기본: 1.0)")
    parser.add_argument("--parent-max-mdd-delta", type=float, default=2.0,
                        help="부모 대비 허용 가능한 MDD 악화 폭 %%p (기본: 2.0)")
    parser.add_argument("--final-eval-env",
                        choices=["baby", "full"], default="full",
                        help="자동 폐기 후 생존 모델 최종 평가에 사용할 환경\n"
                             "full: 본절컷·승률보너스 포함 완전체 환경 (기본)\n"
                             "baby: 초기 배치 평가와 동일한 경량 환경")
    parser.add_argument("--skip-finetune", action="store_true", default=False,
                        help="Phase 1.5(Baby→Full 커리큘럼 파인튜닝)를 건너뜁니다.\n"
                             "기존 Baby-only 진화 파이프라인 동작을 유지하려면 이 플래그를 사용하세요.")
    parser.add_argument("--finetune-timesteps", type=int, default=500_000,
                        help="Phase 1.5 파인튜닝 총 학습 스텝 수 (기본: 500,000)")

    args = parser.parse_args()
    
    # 안전장치: 폐기 수를 1 미만으로 적었을 때의 버그 방지
    if args.auto_discard_top < 1:
        logger.warning("⚠️ [경고] --auto-discard-top 값은 최소 1 이상이어야 합니다. 강제로 1로 조정합니다.")
        args.auto_discard_top = 1
        
    run_evolution_pipeline(args)