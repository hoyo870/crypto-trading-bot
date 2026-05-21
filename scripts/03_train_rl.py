"""
사령관(RL Agent) 훈련 스크립트

단순화 원칙:
  - --count 만큼 반복 (1회 = 1 seed = 1 모델)
  - seed 는 내부에서 0~99999 랜덤 생성 (재현성보다 다양성 우선)
  - --tuning-profile 은 1개만 지정 (multi-profile 은 04_train_rl_batch 가 담당)
  - 생성된 tag 목록을 tags.txt 파일로 저장 (백테스트 입력용)
"""

import os
import sys
import random
import time
import argparse
import gc
import csv
import logging

import pandas as pd

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ── 경로 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

# ── 로깅 ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, "logs", "orchestrator.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Commander.Train")

_custom_log_dir = os.environ.get("CUSTOM_LOG_DIR")
if _custom_log_dir:
    os.makedirs(_custom_log_dir, exist_ok=True)
    logging.getLogger().addHandler(
        logging.FileHandler(os.path.join(_custom_log_dir, "train.log"), encoding='utf-8')
    )

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.envs.trading_env_baby import BabyLeverageTradingEnv as LeverageTradingEnv
from src.utils.platform_utils import configure_torch

configure_torch("cpu")


# ── 신호 신선도 체크 ───────────────────────────────────────────────────────
def _check_signal_freshness(data_path: str) -> None:
    """BASE 모델 .pth 가 신호 CSV보다 최신이면 재추출 경고를 출력합니다."""
    if not os.path.exists(data_path):
        return
    model_dir = os.path.join(ROOT_DIR, "checkpoints", "base_experts")
    pth_files  = ["long_expert.pth", "short_expert.pth", "context_expert.pth"]
    sig_mtime  = os.path.getmtime(data_path)
    stale = [p for p in pth_files
             if os.path.exists(os.path.join(model_dir, p))
             and os.path.getmtime(os.path.join(model_dir, p)) > sig_mtime]
    if stale:
        sig_time   = time.strftime('%Y-%m-%d %H:%M', time.localtime(sig_mtime))
        model_time = time.strftime('%Y-%m-%d %H:%M',
                                   time.localtime(max(
                                       os.path.getmtime(os.path.join(model_dir, p))
                                       for p in stale
                                   )))
        logger.warning(
            f"[WARN] BASE 모델이 신호 파일보다 최신입니다!\n"
            f"  신호 파일: {data_path}  ({sig_time})\n"
            f"  최신 모델: {stale}  ({model_time})\n"
            f"  재추출 권장: python scripts/02_extract_signals.py --symbol BTC_USDT\n"
            f"  (멀티심볼: ETH_USDT / SOL_USDT / XRP_USDT 도 동일하게 실행)"
        )
    else:
        logger.info(f"[OK] 신호 파일 신선도 확인 완료: {os.path.basename(data_path)}")


# ── 하이퍼파라미터 프로파일 ────────────────────────────────────────────────
PPO_TUNING_PROFILES = {
    "stable": {
        "policy_kwargs": dict(net_arch=[256, 256, 128]),
        "learning_rate": 1e-4,
        "ent_coef": 0.02,       # 0.01 → 0.02: 탐색 강화 (local optimum 탈출)
        "vf_coef": 0.5,
        "n_steps": 4096,
        "batch_size": 512,      # 128 → 512: 그라디언트 추정 노이즈 감소
        "n_epochs": 10,
        "gae_lambda": 0.95,
    },
    "balanced": {
        "policy_kwargs": dict(net_arch=[256, 256, 128]),
        "learning_rate": 2e-4,
        "ent_coef": 0.025,      # 0.01 → 0.025: 탐색 강화
        "vf_coef": 0.5,
        "n_steps": 4096,        # 2048 → 4096: 어드밴티지 추정 품질 향상
        "batch_size": 256,      # 64 → 256: 더 안정적인 그라디언트
        "n_epochs": 10,
        "gae_lambda": 0.95,
    },
    "aggressive": {
        "policy_kwargs": dict(net_arch=[256, 256, 128]),
        "learning_rate": 3e-4,
        "ent_coef": 0.03,       # 0.02 → 0.03: 적극적 탐색
        "vf_coef": 0.5,
        "n_steps": 2048,
        "batch_size": 256,      # 64 → 256: 그라디언트 품질 개선
        "n_epochs": 10,
        "gae_lambda": 0.95,
    },
}


# ── SmartStopCallback ──────────────────────────────────────────────────────
class SmartStopCallback(BaseCallback):
    """평가 보상 정체/정책 퇴화를 감지해 학습을 조기 종료합니다."""

    def __init__(self, eval_callback, patience=30, eval_freq=10_000,
                 entropy_threshold=-0.01, reward_target=None,
                 total_timesteps=5_000_000, no_improve_start_ratio=0.1,
                 verbose=1):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.patience = patience
        self.eval_freq = eval_freq
        self.reward_target = reward_target
        self.entropy_threshold = entropy_threshold
        ratio = min(1.0, max(0.1, float(no_improve_start_ratio)))
        self.no_improve_check_start_step = max(1, int(total_timesteps * ratio))
        self._no_improve_count = 0
        self._best_reward = -np.inf

    def _on_step(self) -> bool:
        entropy = self.logger.name_to_value.get("train/entropy_loss", None)
        if entropy is not None and entropy > self.entropy_threshold:
            logger.info(f"[SmartStop] 정책 퇴화 감지: entropy_loss={entropy:.6f} → 종료")
            return False

        if self.n_calls % self.eval_freq != 0:
            return True

        current_best = self.eval_callback.best_mean_reward
        if current_best == -np.inf:
            return True

        if current_best > self._best_reward:
            self._best_reward = current_best
            self._no_improve_count = 0
        else:
            if self.n_calls < self.no_improve_check_start_step:
                return True
            self._no_improve_count += 1

        if self._no_improve_count >= self.patience:
            logger.info(f"[SmartStop] Early Stopping (best={self._best_reward:.2f})")
            return False

        if self.reward_target is not None and self._best_reward >= self.reward_target:
            logger.info(f"[SmartStop] 목표 달성! eval reward={self._best_reward:.2f}")
            return False

        return True


# ── CustomEvalCallback 헬퍼 ──────────────────────────────────────────────
def _safe_mean(infos: list, key: str) -> float:
    """info dict 목록에서 특정 key 평균을 안전하게 반환합니다."""
    vals = []
    for info in infos:
        v = info.get(key)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return float(np.mean(vals)) if vals else 0.0


_EVAL_CSV_FIELDS = [
    "step", "score", "mean_reward", "std_reward", "min_reward",
    "final_balance", "win_rate", "total_trades", "liquidation_count",
]


def _write_eval_metrics(path: str, **kwargs) -> None:
    """eval_metrics.csv 에 한 행을 안전하게 누적 저장합니다."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_EVAL_CSV_FIELDS,
                                    extrasaction="ignore")
            if write_header:
                writer.writeheader()
            row = {
                k: (round(float(v), 6) if isinstance(v, float) else v)
                for k, v in kwargs.items()
            }
            writer.writerow(row)
    except Exception as exc:
        logger.warning(f"[CustomEval] eval_metrics.csv 저장 실패 (무시): {exc}")


# ── CustomEvalCallback ─────────────────────────────────────────────────────
class CustomEvalCallback(EvalCallback):
    """
    안정성 기반 스코어링으로 best_model.zip 승격 기준을 강화한 EvalCallback.

    승격 조건 (단순 max(mean_reward) 대신):
        stability_score = mean_reward - (std_reward × 0.5) + (min_reward × 0.2)

    추가 기능:
    - 매 평가마다 <model_dir>/eval_metrics.csv 에 실전 지표 누적 저장
    - SmartStopCallback 과 호환 (best_mean_reward 속성 유지)
    """

    def __init__(self, eval_env, best_model_save_path, log_path=None,
                 eval_freq=10_000, n_eval_episodes=10,
                 deterministic=True, render=False, verbose=1):
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
        )
        self._best_score: float = -np.inf
        self._metrics_path: str = os.path.join(
            best_model_save_path, "eval_metrics.csv"
        )

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        # ── 1. DummyVecEnv API 로 에피소드 직접 수집 ─────────────────────
        episode_rewards, episode_infos = self._collect_eval_episodes()
        if not episode_rewards:
            logger.warning(
                f"[CustomEval] 수집된 에피소드 없음 (step={self.num_timesteps}) — 평가 건너뜀"
            )
            return True

        mean_reward = float(np.mean(episode_rewards))
        std_reward  = float(np.std(episode_rewards))
        min_reward  = float(np.min(episode_rewards))

        # SmartStopCallback 호환: mean_reward 기준으로도 best_mean_reward 갱신
        self.last_mean_reward = mean_reward
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

        # ── 2. 안정성 스코어 ─────────────────────────────────────────────
        # 단순 max(mean) 대신: 변동성 패널티 + 최악 케이스 보정 반영
        score = mean_reward - (std_reward * 0.5) + (min_reward * 0.2)

        # ── 3. info 집계 (안전한 fallback) ───────────────────────────────
        final_balance = _safe_mean(episode_infos, "final_balance")
        win_rate      = _safe_mean(episode_infos, "win_rate")
        total_trades  = _safe_mean(episode_infos, "total_trades")
        liq_count     = sum(
            1 for i in episode_infos if i.get("liquidated", False)
        )

        # ── 4. Best model 승격 (stability score 기준) ─────────────────────
        if score > self._best_score:
            self._best_score = score
            if self.best_model_save_path is not None:
                os.makedirs(self.best_model_save_path, exist_ok=True)
                self.model.save(
                    os.path.join(self.best_model_save_path, "best_model")
                )
            logger.info(
                f"    [CustomEval] ✅ Best 갱신 "
                f"(score={score:.4f} | mean={mean_reward:.3f} "
                f"std={std_reward:.3f} min={min_reward:.3f} | "
                f"bal={final_balance:.0f} wr={win_rate:.1f}% "
                f"liq={liq_count}/{len(episode_rewards)})"
            )
        else:
            logger.info(
                f"    [CustomEval] score={score:.4f} ≤ best={self._best_score:.4f} "
                f"(mean={mean_reward:.3f} std={std_reward:.3f})"
            )

        # ── 5. eval_metrics.csv 누적 저장 이전에 Tensorboard 로깅 ──
        try:
            self.logger.record("eval/stability_score", score)
            self.logger.record("eval/final_balance",   final_balance)
            self.logger.record("eval/win_rate",        win_rate)
            self.logger.record("eval/total_trades",    int(total_trades))
            self.logger.dump(self.num_timesteps)
        except Exception:
            pass  # 로거 미초기화 시에도 안전하게 동작

        # ── 5. eval_metrics.csv 누적 저장 ────────────────────────────────
        _write_eval_metrics(
            path=self._metrics_path,
            step=self.num_timesteps,
            score=score,
            mean_reward=mean_reward,
            std_reward=std_reward,
            min_reward=min_reward,
            final_balance=final_balance,
            win_rate=win_rate,
            total_trades=int(total_trades),
            liquidation_count=liq_count,
        )
        return True

    def _collect_eval_episodes(self):
        """
        self.eval_env (DummyVecEnv 또는 원시 환경) 에서 n_eval_episodes 실행.
        반환: (list[float] rewards, list[dict] infos)
        """
        rewards: list = []
        infos: list   = []
        n_envs = getattr(self.eval_env, "num_envs", 1)
        current_rewards = np.zeros(n_envs)
        # 무한루프 방지: 에피소드당 최대 30,000스텝 × n_eval_episodes
        max_steps = self.n_eval_episodes * 30_000
        ep_count = 0

        try:
            reset_out = self.eval_env.reset()
            # gymnasium VecEnv 는 (obs, infos) 튜플, 구형은 obs만 반환
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

            for _ in range(max_steps):
                # ── MaskablePPO eval 마스킹 보장 ─────────────────────────
                # Monitor(ActionMasker(env)) 또는 VecEnv 모두 처리
                masks = None
                try:
                    if hasattr(self.eval_env, "env_method"):
                        # DummyVecEnv 경로
                        masks = np.array(self.eval_env.env_method("action_masks"))
                    elif (hasattr(self.eval_env, "env")
                          and hasattr(self.eval_env.env, "action_masks")):
                        # Monitor(ActionMasker(...)) 단일 환경 경로
                        masks = self.eval_env.env.action_masks()[None]  # (1, n_actions)
                except Exception:
                    masks = None  # 마스킹 미지원 환경은 graceful fallback

                actions, _ = self.model.predict(
                    obs, deterministic=self.deterministic, action_masks=masks
                )
                # SB3 VecEnv: (obs, rewards, dones, infos) 또는 5-tuple
                step_out = self.eval_env.step(actions)
                if len(step_out) == 5:
                    obs, rews, terminateds, truncateds, ep_infos = step_out
                    dones = np.logical_or(terminateds, truncateds)
                else:
                    obs, rews, dones, ep_infos = step_out
                current_rewards += rews

                for i, done in enumerate(dones):
                    if done:
                        rewards.append(float(current_rewards[i]))
                        ep_info = (
                            ep_infos[i]
                            if i < len(ep_infos) and isinstance(ep_infos[i], dict)
                            else {}
                        )
                        infos.append(ep_info)
                        current_rewards[i] = 0.0
                        ep_count += 1
                        if ep_count >= self.n_eval_episodes:
                            return rewards, infos
        except Exception as exc:
            logger.warning(f"[CustomEval] 에피소드 수집 오류: {exc}")
        return rewards, infos


# ── 단일 모델 훈련 ─────────────────────────────────────────────────────────
def train_one(seed, model_tag, leverage, tuning_profile, load_model_path,
              data_path, model_dir, log_dir,
              total_timesteps, eval_freq, patience, reward_target,
              entropy_threshold, no_improve_start_ratio,
              mutation_scale=1.0, n_envs=1, multi_symbol_paths=None):
    """seed 1개에 대한 PPO 훈련을 수행하고 저장합니다."""
    start = time.time()
    logger.info(f"  ▶ [{model_tag}] seed={seed} | lev={leverage}x | profile={tuning_profile} | n_envs={n_envs}")

    # hp는 항상 복사본 사용 (프로파일 원본 변경 방지)
    hp = dict(PPO_TUNING_PROFILES[tuning_profile])

    # CSV 1회 로드 → 모든 환경(DummyVecEnv 포함) 에서 재사용 (중복 I/O 방지)
    # multi_symbol_paths 지정 시 각 심볼별 DF 로드 → 환경 분산 (데이터 다양성 확보)
    if multi_symbol_paths:
        dfs = []
        for p in multi_symbol_paths:
            if os.path.exists(p):
                _df = pd.read_csv(p)
                dfs.append(_df)
                logger.info(f"    멀티심볼 로드: {p} ({len(_df):,}행)")
            else:
                logger.warning(f"    [WARN] 신호 파일 없음 (제외): {p}")
        if not dfs:
            raise FileNotFoundError("지정된 멀티심볼 신호 파일을 하나도 찾지 못했습니다.")
        logger.info(f"    멀티심볼 총 {len(dfs)}개 심볼 로드 완료")
    else:
        df_shared = pd.read_csv(data_path)
        logger.info(f"    CSV 로드 완료: {data_path} ({len(df_shared):,}행)")
        dfs = [df_shared]

    def mask_fn(env): return env.action_masks()

    if n_envs > 1:
        # DummyVecEnv: 동일 프로세스 내 N개 환경을 배치 처리
        # n_steps는 SB3에서 "환경당(per-env)" 값: 나누지 않고 유지해야 어드밴티지 추정 품질 보존
        # 총 롤아웃 버퍼 = n_steps × n_envs (자동 확장) → 더 다양한 전환 수집
        logger.info(f"    DummyVecEnv: n_envs={n_envs}, n_steps(per env)={hp['n_steps']}, "
                    f"total_rollout={hp['n_steps'] * n_envs:,}")
        if len(dfs) > 1:
            # 멀티심볼: 심볼별로 n_envs를 균등 배분 (각 환경이 단일 심볼만 사용)
            n_per_sym = max(1, n_envs // len(dfs))
            remainder = n_envs - n_per_sym * len(dfs)
            env_factories = []
            for idx, _df in enumerate(dfs):
                count = n_per_sym + (1 if idx < remainder else 0)
                for _ in range(count):
                    def _make(_d=_df):
                        def _init():
                            e = LeverageTradingEnv(df=_d, leverage=leverage, mode="train")
                            return Monitor(ActionMasker(e, mask_fn))
                        return _init
                    env_factories.append(_make())
            train_env = DummyVecEnv(env_factories)
            logger.info(
                f"    멀티심볼 DummyVecEnv: {len(dfs)}심볼 × ~{n_per_sym}환경 "
                f"= {len(env_factories)}개"
            )
        else:
            def _make_train_env():
                e = LeverageTradingEnv(df=dfs[0], leverage=leverage, mode="train")
                return Monitor(ActionMasker(e, mask_fn))
            train_env = DummyVecEnv([_make_train_env] * n_envs)
    else:
        train_env = Monitor(
            ActionMasker(LeverageTradingEnv(df=dfs[0], leverage=leverage, mode="train"), mask_fn)
        )

    # eval 환경: 첫 번째 심볼(BTC 기준)로 일관된 평가
    def mask_fn_eval(env): return env.action_masks()
    eval_env = Monitor(
        ActionMasker(LeverageTradingEnv(df=dfs[0], leverage=leverage, mode="eval"), mask_fn_eval)
    )

    if load_model_path and os.path.exists(load_model_path):
        logger.info(f"    부모 모델 로드: {load_model_path}")
        # 로컬 RNG 사용 → 전역 NumPy RNG 오염 방지
        rng = np.random.default_rng(seed)
        s = float(np.clip(mutation_scale, 0.0, 1.0))  # 적응형 변이 폭 스케일 (0.0~1.0)

        # 파인튜닝: 변이 폭 50% 완화 (Catastrophic Forgetting 방지, 기존 학습 보존)
        s_original = s
        s = s * 0.5
        logger.info(f"    [파인튜닝 모드] 변이 폭 완화: {s_original:.2f} → {s:.2f}")

        # 1. 엔트로피(ENT): [1-0.2s, 1+0.5s] 범위 변이, 클램프 [0.003, 0.03]
        mutated_ent = float(np.clip(
            hp["ent_coef"] * rng.uniform(1.0 - 0.2 * s, 1.0 + 0.5 * s),
            0.003, 0.03
        ))

        # 2. 학습률(LR): [1-0.2s, 1+0.2s] 범위 변이, 클램프 [1e-5, 5e-4]
        mutated_lr = float(np.clip(
            hp["learning_rate"] * rng.uniform(1.0 - 0.2 * s, 1.0 + 0.2 * s),
            1e-5, 5e-4
        ))

        # 3. vf_coef: ±15%*s 변이, 클램프 [0.3, 0.7]
        mutated_vf = float(np.clip(
            hp["vf_coef"] * rng.uniform(1.0 - 0.15 * s, 1.0 + 0.15 * s),
            0.3, 0.7
        ))

        # 4. clip_range: ±10%*s 변이, 클램프 [0.1, 0.3]
        mutated_clip = float(np.clip(
            0.2 * rng.uniform(1.0 - 0.10 * s, 1.0 + 0.10 * s),
            0.1, 0.3
        ))

        logger.info(
            f"    변이 적용(완화scale={s:.2f}) → "
            f"ent={mutated_ent:.5f}, lr={mutated_lr:.2e}, "
            f"vf={mutated_vf:.3f}, clip={mutated_clip:.3f}"
        )
        model = MaskablePPO.load(
            load_model_path, env=train_env, seed=seed,
            tensorboard_log=log_dir,
            custom_objects={
                "ent_coef":      mutated_ent,
                "learning_rate": mutated_lr,
                "vf_coef":       mutated_vf,
                "clip_range":    mutated_clip,
                "n_steps":       hp["n_steps"],  # n_envs 조정 반영
            }
        )

        # 5. 가우시안 노이즈: 정책 가중치에 상대적 미세 교란 (σ_rel = 0.003*s)
        noise_std = 0.003 * s
        if noise_std > 0:
            with torch.no_grad():
                for param in model.policy.parameters():
                    noise = torch.randn_like(param) * noise_std * param.abs().mean().clamp(min=1e-8)
                    param.add_(noise)
            logger.info(f"    가우시안 노이즈 적용 (σ_rel={noise_std:.4f})")
    else:
        model = MaskablePPO("MlpPolicy", train_env, verbose=0, seed=seed,
                    tensorboard_log=log_dir, **hp)

    eval_cb = CustomEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, model_tag),
        log_path=os.path.join(model_dir, model_tag, "results"),
        eval_freq=eval_freq, deterministic=True, render=False,
    )
    smart_stop = SmartStopCallback(
        eval_callback=eval_cb, patience=patience, eval_freq=eval_freq,
        reward_target=reward_target, entropy_threshold=entropy_threshold,
        total_timesteps=total_timesteps,
        no_improve_start_ratio=no_improve_start_ratio,
    )

    try:
        model.learn(total_timesteps=total_timesteps,
                    callback=[eval_cb, smart_stop],
                    reset_num_timesteps=False,
                    tb_log_name=model_tag)
    except KeyboardInterrupt:
        logger.warning("사용자 중단.")
    finally:
        os.makedirs(os.path.join(model_dir, model_tag), exist_ok=True)
        model.save(os.path.join(model_dir, model_tag, f"final_model_{model_tag}.zip"))
        del model, train_env, eval_env
        gc.collect()

    elapsed = int(time.time() - start)
    logger.info(f"  ✅ [{model_tag}] 완료 ({elapsed//60}분 {elapsed%60}초)")
    return model_tag


# ── 배치 실행 ──────────────────────────────────────────────────────────────
def _make_tag(model_dir, leverage, tuning_profile):
    """lev{N}_{prof}_seed{S}_{idx:03d} 형식의 순차 태그 생성."""
    os.makedirs(model_dir, exist_ok=True)
    prof_code = {"stable": "stb", "balanced": "bal", "aggressive": "agg"}.get(tuning_profile, "unk")
    seed = random.randint(0, 99999)
    prefix = f"lev{leverage}_{prof_code}_seed{seed}"
    # 혹시 같은 seed가 나왔을 때 충돌 방지용 suffix
    idx = 1
    for folder in os.listdir(model_dir):
        if folder.startswith(prefix + "_"):
            suffix = folder[len(prefix) + 1:]
            if suffix.isdigit():
                idx = max(idx, int(suffix) + 1)
    return f"{prefix}_{idx:03d}", seed


def run_train_batch(args):
    tags_created = []
    batch_start = time.time()

    for i in range(args.count):
        tag, seed = _make_tag(args.model_dir, args.leverage, args.tuning_profile)
        train_one(
            seed=seed, model_tag=tag, leverage=args.leverage,
            tuning_profile=args.tuning_profile,
            load_model_path=args.load_model,
            data_path=args.data_path,
            model_dir=args.model_dir, log_dir=args.log_dir,
            total_timesteps=args.timesteps, eval_freq=args.eval_freq,
            patience=args.patience, reward_target=args.reward_target,
            entropy_threshold=args.entropy_threshold,
            no_improve_start_ratio=args.no_improve_start_ratio,
            mutation_scale=args.mutation_scale,
            n_envs=args.n_envs,
            multi_symbol_paths=getattr(args, 'multi_symbol_paths', None),
        )
        tags_created.append(tag)
        total_elapsed = int(time.time() - batch_start)
        logger.info(f"🔄 [{i+1:03d}/{args.count:03d}] 누적 소요: {total_elapsed//60}분")

    # ── 생성된 태그 목록 파일 저장 ──────────────────────────────────────────
    tags_file = os.path.join(args.model_dir, "tags.txt")
    # 기존 목록에 누적 (같은 세대 폴더에 여러 batch 실행 시 병합)
    existing = []
    if os.path.exists(tags_file):
        with open(tags_file, "r", encoding="utf-8") as f:
            existing = [line.strip() for line in f if line.strip()]
    merged = sorted(set(existing + tags_created))
    with open(tags_file, "w", encoding="utf-8") as f:
        f.write("\n".join(merged) + "\n")
    logger.info(f"📄 태그 목록 저장: {tags_file} ({len(merged)}개)")

    return tags_created


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commander RL 단일 프로파일 훈련")

    parser.add_argument("--count",    type=int, default=1,
                        help="훈련할 모델 수 (각각 랜덤 seed)")
    parser.add_argument("--leverage", type=int, default=2,
                        help="레버리지 배수 (기본 2)")
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                        help="총 훈련 타임스텝")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="평가 주기 (steps)")
    parser.add_argument("--tuning-profile",
                        choices=["stable", "balanced", "aggressive"],
                        default="balanced",
                        help="하이퍼파라미터 프로파일 (1개만 지정)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="파인튜닝용 부모 모델 (.zip)")
    parser.add_argument("--patience", type=int, default=30,
                        help="SmartStop 인내심 (평가 주기 단위)")
    parser.add_argument("--reward-target", type=float, default=1e8)
    parser.add_argument("--entropy-threshold", type=float, default=-0.01)
    parser.add_argument("--no-improve-start-ratio", type=float, default=0.25)
    parser.add_argument("--mutation-scale", type=float, default=1.0,
                        help="변이 폭 스케일 (1.0=최대, 0.0=변이 없음; run_evolution.py가 세대별 자동 조정)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="DummyVecEnv 병렬 환경 수 (기본=4). 1=단일 환경.\n"
                             "n_steps를 n_envs로 나눠 유효 버퍼 크기를 유지합니다.")
    parser.add_argument(
        "--multi-symbol", action="store_true", default=False,
        help="BTC/ETH/SOL/XRP 4개 심볼 신호를 모두 사용해 학습합니다 (데이터 다양성 확보).\n"
             "활성화 시 n_envs를 심볼 수로 균등 분배하며, eval은 BTC 기준으로 수행합니다.\n"
             "전제: 각 심볼의 {SYM}_signals_log.csv 가 최신 BASE 모델로 재추출돼 있어야 합니다."
    )

    default_data  = os.path.join(ROOT_DIR, "data", "signals", "base_signals_log.csv")
    default_model = os.path.join(ROOT_DIR, "checkpoints", "rl_generations")
    default_log   = os.path.join(ROOT_DIR, "logs", "train")

    parser.add_argument("--data-path",  type=str, default=default_data)
    parser.add_argument("--model-dir",  type=str, default=default_model)
    parser.add_argument("--log-dir",    type=str, default=default_log)

    args = parser.parse_args()

    # 환경변수 우선 (run_evolution.py / 04_train_rl_batch.py 연동)
    args.model_dir      = os.environ.get("CUSTOM_MODEL_DIR",  args.model_dir)
    args.log_dir        = os.environ.get("CUSTOM_LOG_DIR",    args.log_dir)
    args.mutation_scale = float(os.environ.get("MUTATION_SCALE", args.mutation_scale))

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"데이터 파일 없음: {args.data_path}")

    # ── 멀티심볼 경로 결정 ──────────────────────────────────────────────────
    if args.multi_symbol:
        _symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT"]
        _sig_dir = os.path.join(ROOT_DIR, "data", "signals")
        args.multi_symbol_paths = [
            os.path.join(_sig_dir, f"{sym}_signals_log.csv") for sym in _symbols
        ]
        # eval 기준 파일도 BTC로 동기화
        args.data_path = args.multi_symbol_paths[0]
        logger.info(f"멀티심볼 모드 활성화: {_symbols}")
        for p in args.multi_symbol_paths:
            _check_signal_freshness(p)
    else:
        args.multi_symbol_paths = None
        _check_signal_freshness(args.data_path)

    run_train_batch(args)
