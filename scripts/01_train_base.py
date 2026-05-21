import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from copy import deepcopy
import logging

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in os.sys.path:
    os.sys.path.insert(0, ROOT_DIR)

from src.models.base_models import PriceActionExpert, ContextExpert, prepare_expert_data, _MAX_NEG_RATIO
from src.utils.platform_utils import get_device, configure_torch, log_platform_info, get_optimal_workers, get_pin_memory


class FocalLoss(nn.Module):
    """
    Focal Loss: 쉬운 음성 샘플은 down-weight하고 어려운 양성 샘플에 집중.
    alpha: 양성 클래스 가중치 (pos_weight_raw 기반으로 설정)
    gamma: 집중 강도 (0=BCE, 2=표준 focal)
    BCELoss+Sigmoid 모델과 호환됩니다.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1.0 - eps)
        bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        pt = torch.where(target == 1.0, pred, 1.0 - pred)
        at = torch.where(target == 1.0,
                         torch.full_like(target, self.alpha),
                         torch.ones_like(target))
        return (at * (1.0 - pt) ** self.gamma * bce).mean()

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
logger = logging.getLogger("TrainBase")

def train_expert(expert_type, data_path, seq_length=120, epochs=50, patience=7,
                 extra_data_paths=None, long_split=False):
    start_time = time.time()
    _dev_str = get_device()
    configure_torch(_dev_str)
    device = torch.device(_dev_str)
    log_platform_info(logger)
    logger.info("")
    logger.info(f"{'='*50}")
    logger.info(f"🚀 [{expert_type.upper()} EXPERT] 모델 훈련 시작 (Device: {device})")
    logger.info(f"{'='*50}")

    # 데이터 로드 (기본 심볼 - BTC, val/test 분할 기준)
    train_loader, val_loader, test_loader, input_dim, pos_weight = prepare_expert_data(
        data_path, expert_type, seq_length, long_split=(long_split and expert_type == 'long')
    )

    # [Fix 13] Multi-symbol: 추가 심볼의 train 데이터를 기본 심볼(BTC) train 데이터에 합산
    # val/test는 BTC 기준 유지 → AUC 비교 일관성 보장
    # 모든 피처가 가격 정규화됨(log_return, ratio 등) → 심볼 간 스케일 불일치 없음
    if extra_data_paths:
        _extra_datasets = []
        for _ep in extra_data_paths:
            if not os.path.exists(_ep):
                logger.warning(f"[Multi-symbol] 파일 없음, 건너뜀: {_ep}")
                continue
            _extra_train_loader, _, _, _, _ = prepare_expert_data(_ep, expert_type, seq_length)
            _extra_datasets.append(_extra_train_loader.dataset)
        if _extra_datasets:
            _combined = ConcatDataset([train_loader.dataset] + _extra_datasets)
            _n_extra = sum(len(d) for d in _extra_datasets)
            logger.info(
                f"[Fix 13] Multi-symbol 훈련 데이터 합산: "
                f"BTC({len(train_loader.dataset):,}) + 추가({_n_extra:,}) = 합계({len(_combined):,})"
            )
            train_loader = DataLoader(
                _combined, batch_size=256, shuffle=True,
                num_workers=get_optimal_workers(), pin_memory=get_pin_memory(),
                persistent_workers=(get_optimal_workers() > 0)
            )

    # ── 전문가별 모델 / 하이퍼파라미터 (검증된 최적값) ────────────────────────────
    # LONG   : hd=128, attn=False, dp=0.4, lr=0.001,  smooth=0.05 → best Val AUC 0.5411
    #          권장 실행: --only long  --epochs 100 --patience 20
    # SHORT  : hd=128, attn=True,  dp=0.4, lr=0.0007, smooth=0.05 → best Val AUC 0.5922
    #          권장 실행: --only short --epochs 60  --patience 10
    # CONTEXT: hd=64,  ContextExpert,       lr=0.001,  smooth=0.0  → best Val AUC 0.6001
    #          권장 실행: --only context --epochs 100 --patience 20
    if expert_type == 'long':
        model = PriceActionExpert(input_dim=input_dim, hidden_dim=128, dropout=0.4, use_attention=False).to(device)
        _lr = 0.001
        _smooth_eps = 0.05
    elif expert_type == 'short':
        model = PriceActionExpert(input_dim=input_dim, hidden_dim=128, dropout=0.4, use_attention=True).to(device)
        _lr = 0.0007
        _smooth_eps = 0.05   # label smoothing: 0→0.05, 1→0.95
    else:
        # CONTEXT: hidden_dim=64, dropout=0.3
        model = ContextExpert(input_dim=input_dim, hidden_dim=64, dropout=0.3).to(device)
        _lr = 0.001
        _smooth_eps = 0.0    # label smoothing 미적용

    # 손실 함수: long/short는 FocalLoss(alpha=0.75, gamma=2.0), context는 BCEWithLogitsLoss
    # Focal Loss는 희소한 양성 클래스(long 신호 등)를 더 강하게 학습해 출력 범위를 확장합니다.
    # [Fix 7] long/short/context 모두 BCEWithLogitsLoss 으로 통일
    # pos_weight = min(raw_ratio, _MAX_NEG_RATIO) → 데이터 캐핑 1:_MAX_NEG_RATIO 와 일치시켜 이중 보정 방지
    pos_w_val = min(float(pos_weight), float(_MAX_NEG_RATIO))
    pos_w = torch.tensor([pos_w_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # [Fix 8] 최종 FC 바이어스를 클래스 prior logit 으로 초기화
    # pos_weight = neg/pos → prior_logit = log(pos_rate/(1-pos_rate)) = -log(pos_weight)
    # 초기화 시 logit≈0 에서 기울기가 pos_weight*(0.5-1)*n_pos + 0.5*n_neg = 0 으로
    # 정확히 상쇄되어 학습이 정지하는 문제를 방지합니다.
    _prior_logit = -float(torch.log(pos_w).item())  # sigmoid(prior_logit) = pos_rate
    model.fc[-1].bias.data.fill_(_prior_logit)
    logger.info(
        f"  손실함수: BCEWithLogitsLoss(pos_weight={pos_w.item():.2f}) | raw pos_weight={pos_weight:.2f} "
        f"| prior_logit={_prior_logit:.3f} | lr={_lr} | smooth_eps={_smooth_eps}"
    )
    optimizer = Adam(model.parameters(), lr=_lr, weight_decay=1e-4)
    # ── 스케줄러 설정 ─────────────────────────────────────────────────────────────
    # LONG   : CosineAnnealingLR(T_max=epochs, eta_min=1e-5)
    #          epochs 주기로 LR 코사인 감소 → Val AUC 0.5411 달성 (epochs=100, patience=20)
    #          ReduceLROnPlateau 대비 +0.004, WarmRestarts(T_0=30) 대비 +0.002 우위
    # SHORT  : ReduceLROnPlateau(patience=3, factor=0.5) → Val AUC 0.5922 달성
    # CONTEXT: ReduceLROnPlateau(patience=3, factor=0.5) → Val AUC 0.6001 달성 (epochs=100, patience=20)
    if expert_type == 'long':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    else:
        _sched_patience = 3
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=_sched_patience)

    best_val_auc = -1.0
    best_model_weights = None
    patience_counter = 0

    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                predictions = model(X_batch)
                # [Fix 10] Label smoothing: 0→ε, 1→(1-ε)  (val은 hard label 유지)
                _y_loss = y_batch * (1.0 - 2 * _smooth_eps) + _smooth_eps if _smooth_eps > 0 else y_batch
                loss = criterion(predictions, _y_loss)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # 검증
            model.eval()
            val_loss = 0.0
            val_preds, val_trues, val_probs = [], [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()
                    
                    # [Fix 7] 모든 expert가 logit 출력 → sigmoid로 확률 변환 후 평가
                    probs = torch.sigmoid(predictions)
                    binary_preds = (probs >= 0.5).float()
                    val_preds.extend(binary_preds.cpu().numpy())
                    val_trues.extend(y_batch.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())

            val_f1  = f1_score(val_trues, val_preds, zero_division=0)
            val_acc = accuracy_score(val_trues, val_preds)
            # AUC-ROC: all-ones/all-zeros 퇴화 솔루션이면 ~0.5, 실제 판별력 반영
            try:
                val_auc = roc_auc_score(val_trues, val_probs)
            except ValueError:
                val_auc = 0.0

            # 퇴화 솔루션 감지: all-ones (val_acc≈0.5, F1>0.63) OR all-zeros (F1≈0, acc 높음)
            is_trivial_ones  = (val_acc < 0.515) and (val_f1 > 0.63)
            is_trivial_zeros = (val_f1 < 0.01)  and (val_acc > 0.55)
            if is_trivial_ones:
                trivial_flag = " [⚠️ trivial: all-1 예측]"
            elif is_trivial_zeros:
                trivial_flag = " [⚠️ trivial: all-0 예측]"
            else:
                trivial_flag = ""

            logger.info(
                f"Epoch [{epoch+1:03d}/{epochs}] | Loss: {train_loss/len(train_loader):.4f} "
                f"| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}{trivial_flag}"
            )

            # CosineAnnealingLR(LONG): 인자 없이 step() — epoch마다 LR 갱신
            # ReduceLROnPlateau(SHORT/CONTEXT): val_auc 기준으로 plateau 감지
            if expert_type == 'long':
                scheduler.step()
            else:
                scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_weights = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[INFO] 조기 종료 발동 ({patience} epochs 개선 없음).")
                    break

    except KeyboardInterrupt:
        _interrupted_epoch = locals().get('epoch', -1)
        logger.warning(
            f"[⚠️ INTERRUPTED] {expert_type.upper()} 훈련 중단됨 "
            f"(Epoch {_interrupted_epoch+1}). 현재까지 최고 모델을 저장합니다."
        )

    # 최고 모델 저장 (이전 best AUC보다 낮으면 덮어쓰지 않음 — Fix 17)
    model_dir = os.path.join(ROOT_DIR, "checkpoints", "base_experts")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{expert_type}_expert.pth")
    meta_path = os.path.join(model_dir, f"{expert_type}_expert_meta.json")

    # 이전 best AUC 로드
    prev_best_auc = -1.0
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as _f:
                prev_best_auc = json.load(_f).get('best_val_auc', -1.0)
        except Exception:
            pass

    elapsed = time.time() - start_time
    if best_model_weights is not None:
        if best_val_auc > prev_best_auc:
            torch.save(best_model_weights, save_path)
            with open(meta_path, 'w') as _f:
                json.dump({'best_val_auc': best_val_auc}, _f)
            logger.info(f"✅ [{expert_type.upper()}] 훈련 완료 및 저장 (Best AUC: {best_val_auc:.4f}, 소요시간: {int(elapsed//60)}분 {int(elapsed%60)}초) -> {save_path}")
        else:
            logger.info(f"⚠️  [{expert_type.upper()}] 훈련 완료 (Best AUC: {best_val_auc:.4f} ≤ 이전 best {prev_best_auc:.4f}) — pth 유지, 소요시간: {int(elapsed//60)}분 {int(elapsed%60)}초")
    else:
        logger.warning(f"[⚠️] {expert_type.upper()} 저장할 모델 없음 (훈련 데이터 부족 또는 즉시 중단)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Base Expert 모델 학습")
    parser.add_argument("--symbol", type=str, default="BTC_USDT",
                        help="학습 대상 심볼 (기본: BTC_USDT). 실제 사용 가능한 값: BTC_USDT, ETH_USDT, SOL_USDT, XRP_USDT")
    parser.add_argument("--data-path", type=str, default=None,
                        help="processed CSV 경로 (미지정 시 --symbol 기반 자동 생성)")
    parser.add_argument("--seq-length", type=int, default=120, help="LSTM 시퀀스 길이")
    parser.add_argument("--epochs", type=int, default=None,
                        help="최대 학습 에폭 (미지정 시 전문가별 자동: long=100, short=60, context=60)")
    parser.add_argument("--patience", type=int, default=None,
                        help="조기 종료 인내심 (미지정 시 전문가별 자동: long=20, short=10, context=10)")
    parser.add_argument("--multi-symbol", action="store_true",
                        help="[Fix 13] BTC+ETH+SOL+XRP 4개 심볼 통합 훈련 (레짐 다양성 확보, val은 BTC 유지)")
    parser.add_argument(
        "--start-from",
        type=str,
        choices=["long", "short", "context"],
        default=None,
        help=(
            "이미 저장된 전문가 모델을 건너뛰고 지정한 전문가부터 훈련합니다. "
            "예: --start-from short  (long_expert.pth 이미 존재 시 유용)"
        ),
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["long", "short", "context"],
        default=None,
        help="지정한 전문가한 가지만 훈련 (Fix 19: LONG 단독 훈련 등)",
    )
    parser.add_argument(
        "--long-split",
        action="store_true",
        default=False,
        help=(
            "Long 전문가 전용 데이터 분할 전략 적용.\n"
            "  기본: train=Bull전체, val=Bear초반 → 도메인 시프트(AUC~0.5)\n"
            "  --long-split: train=Bull초중반(~2024-06), val=Bull후반(~2025-06)\n"
            "  → Train/Val 모두 Bull 레짐으로 일치, Long 신호 품질 향상 기대.\n"
            "  Short/Context 에는 무시됩니다."
        ),
    )
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(
        ROOT_DIR, "data", "processed", f"{args.symbol}_processed.csv"
    )
    if not os.path.exists(data_path):
        logger.error(f"[ERROR] 데이터 파일을 찾을 수 없습니다: {data_path}")
    else:
        logger.info(f"학습 대상 심볼: {args.symbol} | 데이터: {data_path}")

        _order = ["long", "short", "context"]
        # [Fix 19] --only 플래그: 단일 전문가만 훈련
        if args.only:
            _order = [args.only]
            logger.info(f"[Fix 19] --only={args.only}: {args.only} 전문가만 훈련")
            _start_idx = 0
        else:
            _start_idx = _order.index(args.start_from) if args.start_from else 0
            if _start_idx > 0:
                logger.info(f"[INFO] --start-from={args.start_from}: {_order[:_start_idx]} 전문가 건너뜀")

        # [Fix 13] Multi-symbol: BTC 기본, ETH+SOL+XRP train 데이터 추가
        _extra_syms = ['ETH_USDT', 'SOL_USDT', 'XRP_USDT'] if args.multi_symbol else []
        _extra_paths = [
            os.path.join(ROOT_DIR, "data", "processed", f"{s}_processed.csv")
            for s in _extra_syms
        ]
        if args.multi_symbol:
            logger.info(f"[Fix 13] Multi-symbol 모드: BTC(primary val) + {_extra_syms}")

        # 전문가별 기본 epochs/patience (검증된 최적값)
        # --epochs / --patience 명시 시 해당 값으로 전체 전문가에 일괄 적용
        _default_epochs   = {'long': 100, 'short': 60, 'context': 100}
        _default_patience = {'long': 20,  'short': 10, 'context': 20}

        for _expert in _order[_start_idx:]:
            # LONG은 multi-symbol 제외: 4배 데이터 시 양성 예측 고착 발생 확인
            # SHORT/CONTEXT는 multi-symbol로 레짐 다양성 확보
            _ep = _extra_paths if (args.multi_symbol and _expert != 'long') else []
            _ep_epochs   = args.epochs   if args.epochs   is not None else _default_epochs[_expert]
            _ep_patience = args.patience if args.patience is not None else _default_patience[_expert]
            train_expert(_expert, data_path, seq_length=args.seq_length, epochs=_ep_epochs,
                         patience=_ep_patience, extra_data_paths=_ep or None,
                         long_split=args.long_split)

        logger.info("\n🎉 모든 Base 전문가 모델 훈련이 완료되었습니다!")