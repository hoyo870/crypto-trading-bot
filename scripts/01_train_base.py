import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from copy import deepcopy
import logging

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in os.sys.path:
    os.sys.path.insert(0, ROOT_DIR)

from src.models.base_models import PriceActionExpert, ContextExpert, prepare_expert_data
from src.utils.platform_utils import get_device, configure_torch, log_platform_info


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

def train_expert(expert_type, data_path, seq_length=120, epochs=50, patience=7):
    start_time = time.time()
    _dev_str = get_device()
    configure_torch(_dev_str)
    device = torch.device(_dev_str)
    log_platform_info(logger)
    logger.info("")
    logger.info(f"{'='*50}")
    logger.info(f"🚀 [{expert_type.upper()} EXPERT] 모델 훈련 시작 (Device: {device})")
    logger.info(f"{'='*50}")

    # 데이터 로드 (다운샘플링 적용됨)
    train_loader, val_loader, test_loader, input_dim, pos_weight = prepare_expert_data(data_path, expert_type, seq_length)

    # 모델 선택
    if expert_type in ['long', 'short']:
        model = PriceActionExpert(hidden_dim=64, dropout=0.3).to(device)
    else:
        model = ContextExpert(input_dim=input_dim, hidden_dim=64, dropout=0.3).to(device)

    # 손실 함수: long/short는 FocalLoss(alpha=pos_weight, gamma=2.0), context는 BCELoss
    # Focal Loss는 희소한 양성 클래스(long 신호 등)를 더 강하게 학습해 출력 범위를 확장합니다.
    if expert_type in ['long', 'short']:
        # 1:3 capped 후에도 자연 neg:pos 비율(≈2:1)이 그대로 남아 alpha=1.0이면
        # 모델이 항상 "신호 없음"만 예측하는 all-zero collapse에 빠짐.
        # → alpha = pos_weight(neg/pos 비율)로 양성 클래스에 가중치 부여.
        # clamp 4.0: 극단적 alpha는 all-ones 수렴 위험이 있으므로 상한 고정.
        focal_alpha = min(float(pos_weight), 4.0)
        criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)
        logger.info(f"  손실함수: FocalLoss(alpha={focal_alpha:.2f}, gamma=2.0) | raw pos_weight={pos_weight:.2f} (1:3 capped, alpha=pos_weight)")
    else:
        criterion = nn.BCELoss()
        logger.info(f"  손실함수: BCELoss | raw pos_weight={pos_weight:.2f}")
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

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
                loss = criterion(predictions, y_batch)
                
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
                    
                    binary_preds = (predictions >= 0.5).float()
                    val_preds.extend(binary_preds.cpu().numpy())
                    val_trues.extend(y_batch.cpu().numpy())
                    val_probs.extend(predictions.cpu().numpy())

            val_f1  = f1_score(val_trues, val_preds, zero_division=0)
            val_acc = accuracy_score(val_trues, val_preds)
            # AUC-ROC: all-ones/all-zeros 퇴화 솔루션이면 ~0.5, 실제 판별력 반영
            try:
                val_auc = roc_auc_score(val_trues, val_probs)
            except ValueError:
                val_auc = 0.0

            # all-ones/all-zeros 퇴화 솔루션 감지 (val Acc≈0.5 & F1≈0.667 패턴)
            is_trivial = (val_acc < 0.515) and (val_f1 > 0.63)
            trivial_flag = " [⚠️ trivial: all-1 예측]" if is_trivial else ""

            logger.info(
                f"Epoch [{epoch+1:03d}/{epochs}] | Loss: {train_loss/len(train_loader):.4f} "
                f"| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}{trivial_flag}"
            )

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
        logger.warning(
            f"[⚠️ INTERRUPTED] {expert_type.upper()} 훈련 중단됨 "
            f"(Epoch {epoch+1}). 현재까지 최고 모델을 저장합니다."
        )

    # 최고 모델 저장
    model_dir = os.path.join(ROOT_DIR, "checkpoints", "base_experts")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{expert_type}_expert.pth")
    if best_model_weights is not None:
        torch.save(best_model_weights, save_path)
    else:
        logger.warning(f"[⚠️] {expert_type.upper()} 저장할 모델 없음 (훈련 데이터 부족 또는 즉시 중단)")
        
    elapsed = time.time() - start_time
    logger.info(f"✅ [{expert_type.upper()}] 훈련 완료 및 저장 (Best AUC: {best_val_auc:.4f}, 소요시간: {int(elapsed//60)}분 {int(elapsed%60)}초) -> {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Base Expert 모델 학습")
    parser.add_argument("--symbol", type=str, default="BTC_USDT",
                        help="학습 대상 심볼 (기본: BTC_USDT). 실제 사용 가능한 값: BTC_USDT, ETH_USDT, SOL_USDT, XRP_USDT")
    parser.add_argument("--data-path", type=str, default=None,
                        help="processed CSV 경로 (미지정 시 --symbol 기반 자동 생성)")
    parser.add_argument("--seq-length", type=int, default=120, help="LSTM 시퀀스 길이")
    parser.add_argument("--epochs", type=int, default=50, help="최대 학습 에폭")
    parser.add_argument("--patience", type=int, default=7, help="조기 종료 인내심")
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
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(
        ROOT_DIR, "data", "processed", f"{args.symbol}_processed.csv"
    )
    if not os.path.exists(data_path):
        logger.error(f"[ERROR] 데이터 파일을 찾을 수 없습니다: {data_path}")
    else:
        logger.info(f"학습 대상 심볼: {args.symbol} | 데이터: {data_path}")

        _order = ["long", "short", "context"]
        _start_idx = _order.index(args.start_from) if args.start_from else 0
        if _start_idx > 0:
            logger.info(f"[INFO] --start-from={args.start_from}: {_order[:_start_idx]} 전문가 건너뜀")

        for _expert in _order[_start_idx:]:
            train_expert(_expert, data_path, seq_length=args.seq_length, epochs=args.epochs, patience=args.patience)

        logger.info("\n🎉 모든 Base 전문가 모델 훈련이 완료되었습니다!")