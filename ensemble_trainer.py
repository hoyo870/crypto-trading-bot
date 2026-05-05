"""
앙상블 모델 학습기: 5개의 독립적인 모델을 다른 시드로 학습
각 모델은 models/ensemble/ 디렉토리에 저장됨
"""
import os
import torch
from copy import deepcopy
from sklearn.metrics import f1_score
from crypto_model_training import (
    MultiBranchCryptoPredictor,
    FocalLoss,
    prepare_data,
    evaluate_model,
)
import warnings
warnings.filterwarnings('ignore')


class EnsembleTrainer:
    def __init__(self, data_filepath, num_models=5, device=None, patience=20):
        """
        Parameters:
        -----------
        data_filepath : str
            학습 데이터 경로 (CSV)
        num_models : int
            앙상블 모델 개수 (기본 5)
        device : torch.device
            학습 장치 (mps/cuda/cpu)
        patience : int
            조기 종료 patience
        """
        self.data_filepath = data_filepath
        self.num_models = num_models
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.patience = patience
        self.trained_models = []
        
        # 앙상블 디렉토리 생성
        os.makedirs("models/ensemble", exist_ok=True)

    def train_ensemble(self):
        """5개 모델을 다른 시드로 학습"""
        print(f"\n{'='*60}")
        print(f"🚀 앙상블 학습 시작 ({self.num_models}개 모델, 시드 0~{self.num_models-1})")
        print(f"{'='*60}\n")

        # 데이터 로드 (모든 모델이 동일하게 사용)
        train_loader, val_loader, test_loader, num_indicators, num_patterns, class_weights = prepare_data(
            self.data_filepath,
            seq_length=120
        )

        for model_id in range(self.num_models):
            print(f"\n[모델 {model_id+1}/{self.num_models}] 학습 중 (시드={model_id})...")
            print("-" * 60)

            # 재현성을 위해 시드 설정
            torch.manual_seed(model_id)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(model_id)

            # 새로운 모델 생성
            model = MultiBranchCryptoPredictor(
                num_indicators=num_indicators,
                num_patterns=num_patterns,
                dropout=0.3
            )

            # Focal Loss로 학습
            criterion = FocalLoss(
                gamma=2.0,
                class_weights=class_weights.to(self.device),
                label_smoothing=0.1
            )
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.001,
                weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3
            )

            # 모델 학습
            model = model.to(self.device)
            best_val_f1 = -1.0
            best_model_weights = None
            patience_counter = 0

            for epoch in range(100):
                # ─ 학습 ─
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # ─ 검증 ─
                model.eval()
                val_loss = 0.0
                val_preds = []
                val_trues = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        predictions = model(X_batch)
                        loss = criterion(predictions, y_batch)
                        val_loss += loss.item()

                        _, predicted = torch.max(predictions.data, 1)
                        val_preds.extend(predicted.cpu().numpy())
                        val_trues.extend(y_batch.cpu().numpy())

                avg_val_loss = val_loss / len(val_loader)
                current_lr = optimizer.param_groups[0]['lr']

                # F1 Score 계산 (Long/Short)
                val_f1 = f1_score(val_trues, val_preds, labels=[1, 2], average='macro', zero_division=0)

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1:03d}/100] | Loss: {avg_train_loss:.4f} | F1: {val_f1:.4f} | LR: {current_lr:.6f}")

                scheduler.step(val_f1)

                # 조기 종료
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_weights = deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"  → {self.patience} Epoch 동안 개선 없음. 조기 종료.")
                        break

            # 최고 성능 모델 로드
            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)

            # 모델 저장
            save_path = f"models/ensemble/model_{model_id}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ 저장 완료: {save_path} (Best F1: {best_val_f1:.4f})")

            self.trained_models.append(model)

            # 테스트 셋 평가
            print(f"  📊 테스트 셋 평가:")
            evaluate_model(model, test_loader, self.device)

        print(f"\n{'='*60}")
        print(f"✅ 앙상블 학습 완료! {self.num_models}개 모델 저장됨")
        print(f"{'='*60}\n")

        return self.trained_models


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    filepath = "data/BTC_USDT_processed.csv"

    if not os.path.exists(filepath):
        print(f"[ERROR] '{filepath}' 파일을 찾을 수 없습니다.")
    else:
        trainer = EnsembleTrainer(filepath, num_models=5, device=device, patience=10)
        trainer.train_ensemble()
        print("[INFO] 앙상블 모델 학습 완료! crypto_backtester.py에서 --use-ensemble 옵션으로 사용 가능합니다.")
