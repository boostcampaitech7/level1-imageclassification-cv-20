'''
cutmix.py version 2
1) cutmix 증강을 배치 단위로 동작하도록 수정
2) 훈련 과정에서 기존 dataset과 cutmix 증강 dataset 을 모두 학습

'''
import os
from typing import Tuple, Any, Callable, List, Optional, Union

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
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                targets_one_hot = F.one_hot(targets, num_classes=config.NUM_CLASSES).float()
                loss = self.loss_fn(outputs, targets_one_hot)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                progress_bar.set_postfix(loss=loss.item(), acc=correct/total)
        
        return total_loss / len(self.val_loader), correct / total


    def train(self) -> None:
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

            self.save_model(epoch, val_loss, val_acc)
            self.scheduler.step()

def get_trainer(train_loader, val_loader) -> Trainer:
    model = get_model()
    return Trainer(model, train_loader, val_loader)


model_name = config.MODEL_NAME
# traindata_dir = config.TRAIN_DATA_DIR
# traindata_info_file = os.path.join(traindata_dir, '../trainDelFlip_objectsplit_train_upWithcanny.csv')
# traindata_info_file = os.path.abspath(traindata_info_file)
save_result_path = config.CHECKPOINT_DIR

# train_info = pd.read_csv(traindata_info_file)

# train_df, val_df = train_test_split(train_info, test_size=1-config.TRAIN_RATIO,
# stratify=train_info['target'],random_state=20)

# 학습에 사용할 Transform을 선언.
transform_selector = TransformSelector(transform_type = 'sketch_albumentations')
train_transform = transform_selector.get_transform(is_train=True)
val_transform = transform_selector.get_transform(is_train=False)

# 데이터셋 로드 (canny)
traindata_dir = "./data/trainDelFlip_objectsplit_train_upWithcanny"
valdata_dir = "./data/trainDelFlip_objectsplit_val"

train_df = pd.read_csv("./data/trainDelFlip_objectsplit_train_upWithcanny.csv")
val_df = pd.read_csv("./data/trainDelFlip_objectsplit_val.csv")

train_dataset = CustomDataset(root_dir=traindata_dir,info_df=train_df,transform=train_transform)
val_dataset = CustomDataset(root_dir=valdata_dir,info_df=val_df,transform=val_transform)

# # 데이터셋 로드
# train_dataset = CustomDataset(root_dir=traindata_dir,info_df=train_df,transform=train_transform)
# val_dataset = CustomDataset(root_dir=traindata_dir,info_df=val_df,transform=val_transform)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

# 모델, 손실 함수, 옵티마이저 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


# 데이터 로더 생성 (CutMixDataset 사용하지 않음)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

# 트레이너 생성 및 학습 시작
trainer = get_trainer(train_loader, val_loader)
trainer.train()

# 모델 추론을 위한 함수
def inference(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader
):
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            predictions.extend(preds.cpu().detach().numpy())
    
    return predictions

# 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
model_name = config.MODEL_NAME
testdata_dir = config.TEST_DATA_DIR
testdata_info_file = os.path.join(testdata_dir, '../test.csv')
testdata_info_file = os.path.abspath(testdata_info_file)
save_result_path = config.CHECKPOINT_DIR

if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

# 추론 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
test_info = pd.read_csv(testdata_info_file)

# 총 class 수.
num_classes = config.NUM_CLASSES

# 추론에 사용할 Transform을 선언.
transform_selector = TransformSelector(
    transform_type = "sketch_albumentations"
)
test_transform = transform_selector.get_transform(is_train=False)

# 추론에 사용할 Dataset을 선언.
test_dataset = CustomDataset(
    root_dir=testdata_dir,
    info_df=test_info,
    transform=test_transform,
    is_inference=True
)

# 추론에 사용할 DataLoader를 선언.
test_loader = DataLoader(
    test_dataset, 
    batch_size=config.BATCH_SIZE, 
    shuffle=False,
    drop_last=False
)


# 추론에 사용할 장비를 선택.
# torch라이브러리에서 gpu를 인식할 경우, cuda로 설정.
device = config.DEVICE if torch.cuda.is_available() else 'cpu'

# 추론에 사용할 Model을 선언.
model_selector = ModelSelector(
    model_type=config.MODEL_TYPE, 
    num_classes=config.NUM_CLASSES,
    model_name=config.MODEL_NAME, 
    pretrained=False
)
model = model_selector.get_model()

# best epoch 모델을 불러오기.
model.load_state_dict(torch.load(os.path.join(save_result_path, "best_model.pt"), map_location=device))

# predictions를 CSV에 저장할 때 형식을 맞춰서 저장
# 테스트 함수 호출
predictions = inference(
    model=model, 
    device=config.DEVICE, 
    test_loader=test_loader
)

# 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
test_info['target'] = predictions
test_info = test_info.reset_index().rename(columns={"index": "ID"})

# DataFrame 저장
test_info.to_csv(f"{config.RESULT_DIR}/{model_name}_{config.EPOCHS}_output.csv", index=False)

