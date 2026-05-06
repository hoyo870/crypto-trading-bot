import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy

from crypto_base_models import PriceActionExpert, ContextExpert, prepare_expert_data

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def train_expert(expert_type, data_path, seq_length=120, epochs=50, patience=7):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"🚀 [{expert_type.upper()} EXPERT] 모델 훈련 시작 (Device: {device})")
    print(f"{'='*50}")

    # 데이터 로드 (다운샘플링 적용됨)
    train_loader, val_loader, test_loader, input_dim = prepare_expert_data(data_path, expert_type, seq_length)

    # 모델 선택
    if expert_type in ['long', 'short']:
        model = PriceActionExpert(hidden_dim=64, dropout=0.3).to(device)
    else:
        model = ContextExpert(input_dim=input_dim, hidden_dim=64, dropout=0.3).to(device)

    # 이진 분류를 위한 손실 함수 (BCELoss)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = -1.0
    best_model_weights = None
    patience_counter = 0

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
        val_preds, val_trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
                
                # 0.5 이상이면 1로 판단
                binary_preds = (predictions >= 0.5).float()
                val_preds.extend(binary_preds.cpu().numpy())
                val_trues.extend(y_batch.cpu().numpy())

        val_f1 = f1_score(val_trues, val_preds, zero_division=0)
        val_acc = accuracy_score(val_trues, val_preds)
        
        print(f"Epoch [{epoch+1:03d}/{epochs}] | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] 조기 종료 발동 ({patience} epochs 개선 없음).")
                break

    # 최고 모델 저장
    model_dir = os.path.join(ROOT_DIR, "models", "commander", "base")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{expert_type}_expert.pth")
    if best_model_weights is not None:
        torch.save(best_model_weights, save_path)
    print(f"✅ [{expert_type.upper()}] 훈련 완료 및 저장 (Best F1: {best_val_f1:.4f}) -> {save_path}")


if __name__ == "__main__":
    data_path = os.path.join(ROOT_DIR, "data", "BTC_USDT_processed.csv")
    if not os.path.exists(data_path):
        print("[ERROR] 데이터 파일을 찾을 수 없습니다.")
    else:
        # 3명의 전문가를 순차적으로 훈련
        train_expert('long', data_path, seq_length=120)
        train_expert('short', data_path, seq_length=120)
        train_expert('context', data_path, seq_length=120)
        print("\n🎉 모든 Base 전문가 모델 훈련이 완료되었습니다!")