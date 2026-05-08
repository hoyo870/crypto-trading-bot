"""Commander RL Commander 학습 스크립트 (통합 레버리지 환경)."""
import argparse
import os
import re
import shutil
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
# 에피소드 통계/로그 일관성을 위해 Monitor 래퍼를 사용합니다.
from stable_baselines3.common.monitor import Monitor

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from crypto_trading_env import LeverageTradingEnv


class SmartStopCallback(BaseCallback):
    """
    3가지 조건으로 자동 학습 종료:
    1) Early Stopping: eval reward가 patience 횟수 연속 개선 없으면 종료
    2) 정책 퇴화 감지: entropy_loss가 임계치보다 높으면 종료
    3) 목표 달성: eval reward가 reward_target 이상이면 종료
    """
    def __init__(self, eval_callback, patience=20, eval_freq=10000,
                 entropy_threshold=-0.01, reward_target=50.0,
                 total_timesteps=5_000_000, no_improve_start_ratio=0.2,
                 verbose=1):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.patience = patience
        self.eval_freq = eval_freq
        self.entropy_threshold = entropy_threshold
        self.reward_target = reward_target
        ratio = min(1.0, max(0.1, float(no_improve_start_ratio)))
        self.no_improve_start_ratio = ratio
        self.no_improve_check_start_step = max(1, int(total_timesteps * ratio))
        self._no_improve_count = 0
        self._best_reward = -np.inf

    def _on_step(self) -> bool:
        entropy = self.logger.name_to_value.get("train/entropy_loss", None)
        if entropy is not None and entropy > self.entropy_threshold:
            if self.verbose:
                print(f"\n[SmartStop] 💀 정책 퇴화 감지: entropy_loss={entropy:.6f} → 종료")
            return False

        if self.n_calls % self.eval_freq != 0:
            return True

        current_best = self.eval_callback.best_mean_reward
        if current_best == -np.inf:
            return True

        if current_best > self._best_reward:
            self._best_reward = current_best
            self._no_improve_count = 0
            if self.verbose:
                print(f"[SmartStop] ✅ 개선됨: best_reward={self._best_reward:.2f}")
        else:
            if self.n_calls < self.no_improve_check_start_step:
                if self.verbose:
                    print(f"[SmartStop] ⏳ 워밍업 구간: no-improve 체크 보류 ({self.n_calls}/{self.no_improve_check_start_step} steps)")
                return True
            self._no_improve_count += 1
            if self.verbose:
                print(f"[SmartStop] ⚠️  개선 없음 {self._no_improve_count}/{self.patience} (best={self._best_reward:.2f})")

        if self._no_improve_count >= self.patience:
            if self.verbose:
                print(f"\n[SmartStop] ⏹  Early Stopping (best={self._best_reward:.2f})")
            return False

        if self._best_reward >= self.reward_target:
            if self.verbose:
                print(f"\n[SmartStop] 🎯 목표 달성! eval reward={self._best_reward:.2f}")
            return False

        return True

def _build_regime_eval_starts(max_steps, eval_window=20_000):
    anchors = [0.0, 0.25, 0.50, 0.75]
    max_start = max(0, max_steps - eval_window)
    starts = [min(max_start, int(max_steps * q)) for q in anchors]
    return sorted(set(starts)) or [0]

def _build_range_eval_starts(start_step, end_step, eval_window=20_000, n_points=4):
    if end_step <= start_step:
        return [max(0, start_step)]
    span = max(1, end_step - start_step)
    max_start = max(start_step, end_step - eval_window)
    if max_start <= start_step:
        return [start_step]
    anchors = np.linspace(0.0, 1.0, num=max(2, n_points))
    starts = [int(start_step + (max_start - start_step) * q) for q in anchors]
    return sorted(set(starts)) or [start_step]

def _resolve_split_ranges(max_steps, split_mode, train_ratio, eval_ratio):
    if split_mode == "none":
        return 0, max_steps, max(0, max_steps - int(max_steps * eval_ratio)), max_steps
    train_end = int(max_steps * train_ratio)
    eval_start = max(train_end + 1, max_steps - int(max_steps * eval_ratio))
    train_start = 0
    eval_end = max_steps
    return train_start, train_end, eval_start, eval_end

class RegimeEvalEnv(LeverageTradingEnv):
    def __init__(self, data_path, eval_starts, eval_window=20_000, leverage=2):
        super().__init__(data_path=data_path, leverage=leverage)
        self.eval_starts = list(eval_starts)
        self.eval_window = eval_window
        self.eval_idx = 0
    def reset(self, seed=None, options=None):
        options = dict(options or {})
        if 'start_step' not in options:
            options['start_step'] = self.eval_starts[self.eval_idx % len(self.eval_starts)]
        if 'max_ep_steps' not in options:
            options['max_ep_steps'] = self.eval_window
        self.eval_idx += 1
        return super().reset(seed=seed, options=options)

class TrainSliceEnv(LeverageTradingEnv):
    def __init__(self, data_path, leverage, train_start, train_end, train_ep_steps):
        super().__init__(data_path=data_path, leverage=leverage)
        self.train_start = max(0, int(train_start))
        self.train_end = max(self.train_start, int(train_end))
        self.train_ep_steps = int(train_ep_steps)
    def reset(self, seed=None, options=None):
        options = dict(options or {})
        max_start = max(self.train_start, self.train_end - self.train_ep_steps)
        start_step = int(self.np_random.integers(self.train_start, max_start + 1))
        options.setdefault("start_step", start_step)
        options.setdefault("max_ep_steps", self.train_ep_steps)
        return super().reset(seed=seed, options=options)

class EvalSliceEnv(LeverageTradingEnv):
    def __init__(self, data_path, eval_starts, eval_window, leverage):
        super().__init__(data_path=data_path, leverage=leverage)
        self.eval_starts = list(eval_starts)
        self.eval_window = int(eval_window)
        self.eval_idx = 0
    def reset(self, seed=None, options=None):
        options = dict(options or {})
        options.setdefault('start_step', self.eval_starts[self.eval_idx % len(self.eval_starts)])
        options.setdefault('max_ep_steps', self.eval_window)
        self.eval_idx += 1
        return super().reset(seed=seed, options=options)

def _normalize_tag(tag):
    clean = re.sub(r"[^a-zA-Z0-9_-]", "", str(tag)).lower()
    return clean or None

def _next_model_tag(candidates_dir, leverage, seed):
    prefix = f"lev{int(leverage)}_seed{int(seed)}"
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{3}})\.zip$")
    max_idx = 0
    if os.path.isdir(candidates_dir):
        for name in os.listdir(candidates_dir):
            m = pattern.fullmatch(name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return f"{prefix}_{max_idx + 1:03d}"

def train_commander(total_timesteps=5_000_000,
                    eval_freq=10_000,
                    patience=30,
                    reward_target=None,
                    entropy_threshold=-0.01,
                    no_improve_start_ratio=0.2,
                    seed=42,
                    model_tag=None,
                    leverage=2,
                    load_model_path=None,
                    improved_hp=False,
                    split_mode="none",
                    train_ratio=0.7,
                    eval_ratio=0.2,
                    train_ep_steps=20_000,
                    eval_window=20_000,
                    model_dir=None,
                    data_path=None):

    print(f"\n{'='*55}")
    print(f"🚀 Commander RL 훈련 시작 (레버리지 {int(leverage)}x)")
    print(f"{'='*55}")

    if data_path is None:
        data_path = os.path.join(ROOT_DIR, "data", "commander", "base_signals_log.csv")
    if not os.path.exists(data_path):
        print("[ERROR] base_signals_log.csv 없음. data 폴더 확인 필요.")
        return

    full_env = LeverageTradingEnv(data_path=data_path, leverage=leverage)
    max_steps = full_env.max_steps
    train_start, train_end, eval_start, eval_end = _resolve_split_ranges(
        max_steps=max_steps,
        split_mode=split_mode,
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
    )

    if (train_end - train_start) < max(1_000, train_ep_steps) or (eval_end - eval_start) < max(1_000, eval_window):
        print("[WARN] 데이터 분할 구간이 너무 작아 split_mode=none 으로 대체합니다.")
        split_mode = "none"
        train_start, train_end, eval_start, eval_end = _resolve_split_ranges(
            max_steps=max_steps,
            split_mode=split_mode,
            train_ratio=train_ratio,
            eval_ratio=eval_ratio,
        )

    if split_mode == "none":
        raw_env = LeverageTradingEnv(data_path=data_path, leverage=leverage)
        eval_starts = _build_regime_eval_starts(max_steps=raw_env.max_steps, eval_window=eval_window)
        raw_eval_env = RegimeEvalEnv(data_path=data_path, eval_starts=eval_starts,
                                     eval_window=eval_window, leverage=leverage)
        # 학습/평가 환경 모두 Monitor로 감싸서 안정적으로 로그를 수집합니다.
        env = Monitor(raw_env)
        eval_env = Monitor(raw_eval_env)
        print(f"[INFO] 데이터 분할: none (전체 구간)")
        print(f"[INFO] 평가 시작점(국면 분할): {eval_starts}")
    else:
        raw_env = TrainSliceEnv(
            data_path=data_path,
            leverage=leverage,
            train_start=train_start,
            train_end=train_end,
            train_ep_steps=train_ep_steps,
        )
        eval_starts = _build_range_eval_starts(
            start_step=eval_start,
            end_step=eval_end,
            eval_window=eval_window,
            n_points=4,
        )
        raw_eval_env = EvalSliceEnv(
            data_path=data_path,
            eval_starts=eval_starts,
            eval_window=eval_window,
            leverage=leverage,
        )
        # 학습/평가 환경 모두 Monitor로 감싸서 안정적으로 로그를 수집합니다.
        env = Monitor(raw_env)
        eval_env = Monitor(raw_eval_env)
        dt_index = pd.to_datetime(pd.read_csv(data_path, usecols=["datetime"])["datetime"], errors="coerce")
        def _d(step):
            if 0 <= step < len(dt_index) and pd.notna(dt_index.iloc[step]):
                return str(dt_index.iloc[step])
            return "N/A"
        print(f"[INFO] 데이터 분할: holdout")
        print(f"[INFO] train range: {train_start}..{train_end} ({_d(train_start)} ~ {_d(train_end)})")
        print(f"[INFO] eval range : {eval_start}..{eval_end} ({_d(eval_start)} ~ {_d(eval_end)})")
        print(f"[INFO] eval 시작점: {eval_starts}")

    if model_dir is None:
        model_dir = os.path.join(ROOT_DIR, "models", "commander")
    runs_dir = os.path.join(model_dir, "runs")
    candidates_dir = os.path.join(model_dir, "candidates")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(candidates_dir, exist_ok=True)

    requested_tag = _normalize_tag(model_tag) if model_tag else None
    final_tag = requested_tag or _next_model_tag(candidates_dir, leverage=leverage, seed=seed)
    run_dir = os.path.join(runs_dir, final_tag)
    os.makedirs(run_dir, exist_ok=True)

    # ── 모델 초기화 ──────────────────────────────────────────────
    if load_model_path:
        if not os.path.exists(load_model_path):
            print(f"[ERROR] 로드할 모델 파일 없음: {load_model_path}")
            return
        print(f"[INFO] 🔄 파인튜닝 모드: {load_model_path} 로드")
        model = PPO.load(load_model_path, env=env, device="auto",
                         custom_objects={
                             "learning_rate": 1e-4,
                             "ent_coef": 0.005,
                             "n_steps": 4096,
                             "batch_size": 128,
                         })
        print(f"[INFO] 파인튜닝 hp: lr=1e-4, ent_coef=0.005, n_steps=4096, batch=128")
    elif improved_hp:
        policy_kwargs = dict(net_arch=[256, 256, 128])
        print(f"[INFO] 🆕 개선 hp 모드: net_arch=[256,256,128], lr=1e-4")
        model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                    learning_rate=1e-4,
                    ent_coef=0.005,
                    vf_coef=0.5,
                    n_steps=4096,
                    batch_size=128,
                    seed=seed)
    else:
        policy_kwargs = dict(net_arch=[256, 128])
        print(f"[INFO] 기본 hp 모드: net_arch=[256,128], lr=3e-4")
        model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                    learning_rate=3e-4,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    n_steps=2048,
                    batch_size=64,
                    seed=seed)

    eval_callback = EvalCallback(eval_env, best_model_save_path=run_dir,
                                 log_path=run_dir, eval_freq=eval_freq,
                                 deterministic=True, render=False,
                                 n_eval_episodes=len(eval_starts))

    smart_stop = SmartStopCallback(
        eval_callback=eval_callback,
        patience=patience,
        eval_freq=eval_freq,
        entropy_threshold=entropy_threshold,
        reward_target=(float("inf") if reward_target is None else reward_target),
        total_timesteps=total_timesteps,
        no_improve_start_ratio=no_improve_start_ratio,
    )

    print(f"[INFO] 모델 태그: {final_tag} | 레버리지: {int(leverage)}x")
    print("[INFO] 학습 시작...")
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, smart_stop])

    best_model_in_run = os.path.join(run_dir, "best_model.zip")
    if os.path.exists(best_model_in_run):
        promoted_path = os.path.join(candidates_dir, f"{final_tag}.zip")
        shutil.copy2(best_model_in_run, promoted_path)
        shutil.copy2(best_model_in_run, os.path.join(model_dir, "best_model.zip"))
        print(f"[INFO] 후보 모델 저장: {promoted_path}")
        print(f"[INFO] 기본 모델 갱신: {os.path.join(model_dir, 'best_model.zip')}")
    else:
        print("[WARN] best_model.zip 미생성 — eval 미도달 가능성 있음")

    print(f"\n🎉 훈련 완료. 결과 폴더: {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Commander RL (Leverage)")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--entropy-threshold", type=float, default=-0.01)
    parser.add_argument("--no-improve-start-ratio", type=float, default=0.1,
                        help="patience no-improve 체크 시작 비율 (최소 0.1, 최대 1.0)")
    parser.add_argument("--reward-target", type=float, default=1e9,
                        help="목표 eval reward (매우 크게 설정 시 사실상 비활성)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leverage", type=int, default=2,
                        help="레버리지 배수 (1=현물, 2=기본, 3 등)")
    parser.add_argument("--tag", type=str, default=None,
                        help="모델 태그 (기본: 자동 lev{레버리지}_seed{시드}_NNN)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="파인튜닝할 기존 commander 모델 ZIP 경로")
    parser.add_argument("--improved-hp", action="store_true",
                        help="개선 하이퍼파라미터 (net_arch 확장·lr 감소)")
    parser.add_argument("--split-mode", type=str, choices=["none", "holdout"], default="none",
                        help="학습/평가 데이터 분할 모드")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="holdout 모드 학습 비율 (0~1)")
    parser.add_argument("--eval-ratio", type=float, default=0.2,
                        help="holdout 모드 평가 비율 (0~1)")
    parser.add_argument("--train-ep-steps", type=int, default=20_000,
                        help="학습 에피소드 길이")
    parser.add_argument("--eval-window", type=int, default=20_000,
                        help="평가 에피소드 길이")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="모델 저장 루트 디렉토리 (기본: root/models/commander)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="학습/평가 입력 데이터 CSV 경로 (기본: root/data/commander/base_signals_log.csv)")
    args = parser.parse_args()

    reward_target = args.reward_target
    if reward_target >= 1e8:
        reward_target = None

    train_commander(
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        patience=args.patience,
        reward_target=reward_target,
        entropy_threshold=args.entropy_threshold,
        no_improve_start_ratio=args.no_improve_start_ratio,
        seed=args.seed,
        model_tag=args.tag,
        leverage=args.leverage,
        load_model_path=args.load_model,
        improved_hp=args.improved_hp,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        train_ep_steps=args.train_ep_steps,
        eval_window=args.eval_window,
        model_dir=args.model_dir,
        data_path=args.data_path,
    )