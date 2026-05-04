import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

# 1. 시계열 윈도우 생성을 위한 커스텀 데이터셋 클래스
class CryptoTimeSeriesDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        # seq_length(예: 60)만큼의 과거 데이터를 윈도우로 묶음
        x = self.features[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 2. LSTM 딥러닝 모델 아키텍처
class CryptoPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(CryptoPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # M1/M2 등에서 고속 연산이 가능한 LSTM 레이어
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        
        # 출력 레이어 (상승/하락 이진 분류를 위한 1개 노드)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # 시계열의 가장 마지막 시점(현재)의 은닉 상태만 사용하여 미래를 예측
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.squeeze() # (batch_size,)

# 3. 데이터 로딩 및 정답지(Label) 생성 로직
def prepare_data(filepath, seq_length=60):
    print(f"[INFO] 학습 데이터 로드 중: {filepath}")
    df = pd.read_csv(filepath)
    
    # [타겟 라벨링] 6봉(30분) 뒤 종가가 현재 종가보다 높으면 1(상승), 아니면 0(하락)
    df['Target'] = (df['close'].shift(-6) > df['close']).astype(int)
    
    # 미래 데이터를 당겨오면서 발생한 맨 마지막 6개의 결측치 행 제거
    df.dropna(inplace=True) 
    
    # 모델 학습에 불필요한 날짜/타겟 컬럼 제외하고 피처만 추출
    exclude_cols = ['timestamp', 'datetime', 'Target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    features = df[feature_cols].values
    targets = df['Target'].values
    
    # [데이터셋 분리] Time-Series Split (미래 데이터 섞임 방지를 위해 순서 유지)
    n = len(df)
    train_end = int(n * 0.70)  # 70% Train
    val_end = int(n * 0.85)    # 15% Validation
    
    X_train, y_train = features[:train_end], targets[:train_end]
    X_val, y_val = features[train_end:val_end], targets[train_end:val_end]
    X_test, y_test = features[val_end:], targets[val_end:]
    
    # DataLoader 생성 (배치 사이즈는 M1 Max 메모리를 고려하여 256으로 설정)
    train_dataset = CryptoTimeSeriesDataset(X_train, y_train, seq_length)
    val_dataset = CryptoTimeSeriesDataset(X_val, y_val, seq_length)
    test_dataset = CryptoTimeSeriesDataset(X_test, y_test, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print(f"[INFO] 분할 완료 - Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, len(feature_cols)

# 4. 학습 루프 및 조기 종료 (Early Stopping) 관리 클래스
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device, patience=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # BCEWithLogitsLoss: 내부적으로 Sigmoid를 포함하여 수치적으로 더 안정적인 손실 함수
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.patience = patience
        
    def train(self, epochs=50):
        print(f"[INFO] 하드웨어 가속기({self.device}) 기반 학습 시작...")
        best_val_loss = float('inf')
        best_model_weights = None
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            
            # 검증 (Validation) 루프
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    predictions = self.model(X_batch)
                    loss = self.criterion(predictions, y_batch)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(self.val_loader)
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # 조기 종료 (Early Stopping) 체크
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"[INFO] {self.patience} Epoch 동안 검증 손실이 개선되지 않아 조기 종료(Early Stop)합니다.")
                    break
        
        # 최고 성능을 기록했던 모델의 가중치로 복원하여 반환
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
        return self.model

# 5. 최종 모델 평가 함수
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            # 예측값(Logits)을 0~1 사이의 확률로 변환
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
    accuracy = 100 * correct / total
    print(f"\n[RESULT] ⭐ 최종 테스트 데이터셋 예측 정확도 (Accuracy): {accuracy:.2f}%")

if __name__ == "__main__":
    # Mac M1/M2/M3 하드웨어 가속기(MPS) 연결
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1단계 파이프라인에서 추출된 파일명 지정
    filepath = "BTC_USDT_5m_raw.csv"
    
    # 파일이 존재하는지 검증
    if not os.path.exists(filepath):
        print(f"[ERROR] '{filepath}' 파일을 찾을 수 없습니다. 파이프라인 수집이 완료되었는지 확인해 주세요.")
    else:
        # 데이터 준비 (과거 60개 캔들 = 5시간의 흐름을 보고 분석)
        train_loader, val_loader, test_loader, input_size = prepare_data(filepath, seq_length=60)
        
        # 모델 세팅
        model = CryptoPredictorLSTM(input_size=input_size, hidden_size=128, num_layers=2)
        
        # 학습 시작
        trainer = ModelTrainer(model, train_loader, val_loader, device, patience=5)
        trained_model = trainer.train(epochs=50)
        
        # 성과 측정
        evaluate_model(trained_model, test_loader, device)
        
        # 최종 실전 투입용 모델 파일 저장
        os.makedirs("models", exist_ok=True)
        save_path = "models/best_lstm_btc_5m.pth"
        torch.save(trained_model.state_dict(), save_path)
        print(f"[INFO] 💾 최고 성능의 모델 가중치가 안전하게 저장되었습니다: {save_path}")