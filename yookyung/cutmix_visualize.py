import os
import cv2
import timm
import torch
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2

from torchvision.transforms import v2
from torch.nn.functional import one_hot
from src.trainer import DataLoader, get_trainer
from src.transform import TransformSelector

from config import config
from models.models import get_model, ModelSelector
from models.loss import get_loss_function
from models.optimizer import get_optimizer
from models.scheduler import get_scheduler
from src.dataset import CustomDataset

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
            _, _, path_to_remove = self.best_models.pop(0)  # 가장 낮은 정확도 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 높은 정확도의 모델 저장
        if acc > self.high_acc:
            self.high_acc = acc
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Accuracy = {acc:.4f}")

    def train_epoch(self) -> float:
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            batch_size = images.shape[0]  # 현재 배치 크기

            # 원-핫 인코딩
            targets_one_hot = F.one_hot(targets, num_classes=config.NUM_CLASSES).float()

            # 원본 데이터로 학습
            outputs = self.model(images)
            loss_original = self.loss_fn(outputs, targets)

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

# CutMix 함수 (이전과 동일)
def cutmix_batch(images, labels, alpha=1.0):
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)
    
    rand_index = torch.randperm(batch_size)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    
    new_images = images.clone()
    new_images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    new_labels = lam * labels + (1 - lam) * labels[rand_index]
    
    return new_images, new_labels

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# 예시: 배치 크기가 4인 경우의 CutMix 결과 시각화
import matplotlib.pyplot as plt

def visualize_cutmix(images, mixed_images, labels, mixed_labels):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(4):
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[0, i].set_title(f"Original: {labels[i]}")
        axes[1, i].imshow(mixed_images[i].permute(1, 2, 0))
        axes[1, i].set_title(f"Mixed: {mixed_labels[i].argmax().item()}")
    plt.tight_layout()
    plt.show()

# 가상의 배치 데이터 생성
batch_size = 4
images = torch.rand(batch_size, 3, 224, 224)
labels = torch.randint(0, 10, (batch_size,))
labels_one_hot = F.one_hot(labels, num_classes=10).float()

# CutMix 적용
mixed_images, mixed_labels = cutmix_batch(images, labels_one_hot)

# 결과 시각화
visualize_cutmix(images, mixed_images, labels, mixed_labels)