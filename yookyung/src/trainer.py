import os
from typing import Tuple, Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

from config import config
from models.models import get_model
from models.loss import get_loss_function
from models.optimizer import get_optimizer
from models.scheduler import get_scheduler

from sklearn.model_selection import StratifiedKFold
import numpy as np

# def perform_k_fold_cross_validation(train_info, config, get_trainer):
#     # StratifiedKFold 객체 생성
#     skf = StratifiedKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)

#     # 결과를 저장할 리스트
#     fold_results = []

#     # K-Fold 교차 검증 수행
#     for fold, (train_idx, val_idx) in enumerate(skf.split(train_info, train_info['target']), 1):
#         print(f"Fold {fold}/{config.K_FOLDS}")

#         # 훈련 및 검증 데이터 분리
#         train_df = train_info.iloc[train_idx]
#         val_df = train_info.iloc[val_idx]

#         # 데이터 로더 생성 (이 부분은 당신의 데이터 로딩 로직에 맞게 수정해야 합니다)
#         train_loader = create_data_loader(train_df, config.TRAIN_BATCH_SIZE, shuffle=True)
#         val_loader = create_data_loader(val_df, config.VAL_BATCH_SIZE, shuffle=False)

#         # 트레이너 생성 및 훈련 수행
#         trainer = get_trainer(train_loader, val_loader)
#         trainer.train()

#         # 최종 검증 정확도 저장
#         fold_results.append(trainer.high_acc)

#         print(f"Fold {fold} completed. Best validation accuracy: {trainer.high_acc:.4f}")
#         print("-" * 50)

#     # 전체 결과 출력
#     print("\nK-Fold Cross Validation Results:")
#     for fold, acc in enumerate(fold_results, 1):
#         print(f"Fold {fold}: {acc:.4f}")
#     print(f"Average validation accuracy: {np.mean(fold_results):.4f}")
#     print(f"Standard deviation: {np.std(fold_results):.4f}")

# # 사용 예시
# if __name__ == "__main__":
#     perform_k_fold_cross_validation(train_info, config, get_trainer)


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model.to(config.DEVICE)  # 훈련할 모델
        self.device = config.DEVICE  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = get_optimizer(self.model.parameters())  # 최적화 알고리즘
        self.scheduler = get_scheduler(config.SCHEDULER, self.optimizer, steps_per_epoch=len(train_loader)) # 학습률 스케줄러
        self.loss_fn = get_loss_function()  # 손실 함수
        self.epochs = config.EPOCHS  # 총 훈련 에폭 수
        self.result_path = config.CHECKPOINT_DIR # 모델 저장 경로
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf') # 가장 낮은 Loss를 저장할 변수
        self.high_acc = 0.0

    def save_model(self, epoch, loss):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")

        # # 최상위 3개 모델 관리
        # self.best_models.append((acc, epoch, current_model_path))
        # self.best_models.sort()
        # if len(self.best_models) > 3:
        #     _, _, path_to_remove = self.best_models.pop(0)  # 가장 낮은 정확도 모델 삭제
        #     if os.path.exists(path_to_remove):
        #         os.remove(path_to_remove)

        # # 가장 낮은 손실의 모델 저장
        # if acc > self.high_acc:
        #     self.high_acc = acc
        #     best_model_path = os.path.join(self.result_path, 'best_model.pt')
        #     torch.save(self.model.state_dict(), best_model_path)
        #     print(f"Save {epoch}epoch result. Accuracy = {acc:.4f}")

    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            progress_bar.set_postfix(loss=loss.item(), acc=correct/total)
        
        return total_loss / len(self.train_loader), correct / total

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)    
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data,1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                progress_bar.set_postfix(loss=loss.item(), acc=correct/total)
        
        return total_loss / len(self.train_loader), correct / total

    def train(self) -> None:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

            self.save_model(epoch, val_loss)
            self.scheduler.step()


def get_trainer(train_loader, val_loader) -> Trainer:
    model = get_model()
    return Trainer(model, train_loader, val_loader)
