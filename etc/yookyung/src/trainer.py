import os
from typing import Tuple, Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

from config import config
from models.models import get_model
from models.loss import get_loss_function
from models.optimizer import get_optimizer
from models.scheduler import get_scheduler

from utils import cutmix_batch


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
        self.high_acc = 0.0 # 가장 높은 Accuracy를 저장할 변수

    def save_model(self, epoch, loss, acc):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}_acc_{acc:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((acc, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(0)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 높은 정확도의 모델 저장
        if acc >  self.lowest_loss:
            self.high_acc = acc
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Accuracy = {acc:.4f}")
 
    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            if not config.IS_CUTMIX: # cutmix 적용 안하는 경우
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            else: # cutmix 적용 하는 경우 
                targets_one_hot = F.one_hot(targets, num_classes=config.NUM_CLASSES).float()

                # 원본 데이터로 학습
                outputs = self.model(images)
                loss_original = self.loss_fn(outputs, targets_one_hot)

                # CutMix 적용
                images_mixed, targets_mixed = cutmix_batch(images, targets_one_hot)
                outputs_mixed = self.model(images_mixed)
                loss_mixed = self.loss_fn(outputs_mixed, targets_mixed)

                # 두 손실을 합침
                loss = (loss_original + loss_mixed) / 2


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 원본 데이터에 대한 정확도 계산
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

                if config.IS_CUTMIX:
                    targets = F.one_hot(targets, num_classes=config.NUM_CLASSES).float() 
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
