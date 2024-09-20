# 필요 library들을 import합니다.
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
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from transformers import CLIPProcessor
from PIL import Image
import matplotlib.pyplot as plt
import wandb

dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
prompt_path = dir+"/data/zeroshot_classification_templates.txt"

def get_imagenet_ditction(mini=True,values=True):
    if mini:
        if values:
            return list(mini_imagenet_cls_map.values())
        else:
            return mini_imagenet_cls_map
    else:
        if values:
            return list(imagenet_cls_map.values())
        else:
            return imagenet_cls_map
        
class CLIPDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable,
        is_inference: bool = False,
        processor = CLIPProcessor,
        device =torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_print = False
    ):
        # 데이터셋의 기본 경로, 이미지 변환 방법, 이미지 경로 및 레이블을 초기화합니다.
        self.root_dir = root_dir  # 이미지 파일들이 저장된 기본 디렉토리
        self.transform = transform  # 이미지에 적용될 변환 처리
        self.is_inference = is_inference # 추론인지 확인
        self.image_paths = info_df['image_path'].tolist()  # 이미지 파일 경로 목록
        self.processor = processor
        self.device = device
        self.use_print = use_print
        if not self.is_inference:
            self.targets = info_df['target'].tolist()  # 각 이미지에 대한 레이블 목록
            
    def get_images(self,ranges:int,rangee:int):
        images = [Image.open(os.path.join(self.root_dir, self.image_paths[index]) ) for index in range(ranges,rangee)]
        classes = [self.image_paths[index].split("/")[0] for index in range(ranges,rangee)]
        return images,classes
    
    def __len__(self) -> int:
        # 데이터셋의 총 이미지 수를 반환합니다.
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        # 주어진 인덱스에 해당하는 이미지를 로드하고 변환을 적용한 후, 이미지와 레이블을 반환합니다.
        img_path = os.path.join(self.root_dir, self.image_paths[index])  # 이미지 경로 조합
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 이미지를 BGR 컬러 포맷의 numpy array로 읽어옵니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 포맷을 RGB 포맷으로 변환합니다.
        image = self.transform(image)  # 설정된 이미지 변환을 적용합니다.
        # image = Image.open(os.path.join(self.root_dir, self.image_paths[index]) )
        classes = self.image_paths[index].split("/")[0]
        text = mini_imagenet_cls_map[classes]
    
        
        if self.use_print:
            print(str(text))
        if self.is_inference:
            return self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        else:            
            return image,text#self.processor(text=text, images=image, return_tensors="pt", padding=True).to(self.device)
        
class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable,
        is_inference: bool = False
    ):
        # 데이터셋의 기본 경로, 이미지 변환 방법, 이미지 경로 및 레이블을 초기화합니다.
        self.root_dir = root_dir  # 이미지 파일들이 저장된 기본 디렉토리
        self.transform = transform  # 이미지에 적용될 변환 처리
        self.is_inference = is_inference # 추론인지 확인
        self.image_paths = info_df['image_path'].tolist()  # 이미지 파일 경로 목록
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()  # 각 이미지에 대한 레이블 목록

    def __len__(self) -> int:
        # 데이터셋의 총 이미지 수를 반환합니다.
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        # 주어진 인덱스에 해당하는 이미지를 로드하고 변환을 적용한 후, 이미지와 레이블을 반환합니다.
        img_path = os.path.join(self.root_dir, self.image_paths[index])  # 이미지 경로 조합
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 이미지를 BGR 컬러 포맷의 numpy array로 읽어옵니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 포맷을 RGB 포맷으로 변환합니다.
        image = self.transform(image=image)['image']   # 설정된 이미지 변환을 적용합니다.

        if self.is_inference:
            return image
        else:
            target = self.targets[index]  # 해당 이미지의 레이블
            return image, target  # 변환된 이미지와 레이블을 튜플 형태로 반환합니다. 
        
class TorchvisionTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 텐서 변환, 정규화
        common_transforms = [
            transforms.Resize((224, 224)),  # 이미지를 224x224 크기로 리사이즈
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 색상 조정 추가
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    transforms.RandomRotation(15),  # 최대 15도 회전
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기 및 대비 조정
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)  # numpy 배열을 PIL 이미지로 변환
        
        transformed = self.transform(image)  # 설정된 변환을 적용
        
        return transformed  # 변환된 이미지 반환
    
class AlbumentationsTransformTest:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            #A.Normalize(mean=(0, 0, 0), std=(255, 255, 255)),#A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.Rotate(limit=15),  # 최대 15도 회전
                    A.RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 무작위 조정
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환    

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 색상 조정 추가
            self.transform = A.Compose(
                [
                    # Geometric transformations
                    A.Rotate(limit=10, p=0.5),
                    A.Affine(scale=(0.8, 1.2), shear=(-10, 10), p=0.5),
                    A.ElasticTransform(alpha=1, sigma=10, p=0.5),
                    
                    # # Morphological transformations
                    # A.Erosion(kernel=(1, 2), p=0.5),
                    # A.Dilation(kernel=(1, 2), p=0.5),

                    # Noise and blur
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),                

                    # Sketch-specific augmentations
                    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=255, p=0.5),

                    # Advanced techniques
                    A.OneOf([
                        # A.AutoContrast(p=0.5),
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                    ], p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환
    
class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type: str):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["torchvision", "albumentations","AlbumentationsTransformTest"]:
            self.transform_type = transform_type
        
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        
        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        if self.transform_type == 'torchvision':
            transform = TorchvisionTransform(is_train=is_train)
        
        elif self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(is_train=is_train)

        elif self.transform_type == 'AlbumentationsTransformTest':
            transform = AlbumentationsTransformTest(is_train=is_train)
    
        return transform
    
class SimpleCNN(nn.Module):
    """
    간단한 CNN 아키텍처를 정의하는 클래스.
    """
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 순전파 함수 정의
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class TorchvisionModel(nn.Module):
    """
    Torchvision에서 제공하는 사전 훈련된 모델을 사용하는 클래스.
    """
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        pretrained: bool
    ):
        super(TorchvisionModel, self).__init__()
        self.model = models.__dict__[model_name](pretrained=pretrained)
        
        # 모델의 최종 분류기 부분을 사용자 정의 클래스 수에 맞게 조정
        if 'fc' in dir(self.model):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        elif 'classifier' in dir(self.model):
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
    
class TimmModel(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    """
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        pretrained: bool
    ):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
    
class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스.
    """
    def __init__(
        self, 
        model_type: str, 
        num_classes: int, 
        **kwargs
    ):
        
        # 모델 유형에 따라 적절한 모델 객체를 생성
        if model_type == 'simple':
            self.model = SimpleCNN(num_classes=num_classes)
        
        elif model_type == 'torchvision':
            self.model = TorchvisionModel(num_classes=num_classes, **kwargs)
        
        elif model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:

        # 생성된 모델 객체 반환
        return self.model

class Loss(nn.Module):
    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        return self.loss_fn(outputs, targets)
    
class BaseTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str,
        model_name: str,
        patience: int = 5  # 조기 종료를 위한 patience 설정
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model  # 훈련할 모델
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수
        self.result_path = result_path  # 모델 저장 경로
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.highest_val_acc = 1e-10
        self.patience = patience  # 조기 종료를 위한 patience
        self.early_stop_counter = 0
        self.model_name = model_name
        self.pre_best = "/1298y192eh72e1y2"
        # 학습 기록 저장을 위한 리스트
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def accuracy_fn(self, outputs, targets):
        """정확도 계산 함수"""
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == targets)
        accuracy = corrects.double() / len(targets)
        return accuracy.item()
    
    def save_model(self, epoch, acc, loss):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path+"/"+self.model_name, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path+"/"+self.model_name, f'model_epoch_{epoch}_Acc_{acc:.4f}_Loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((acc, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(1)  # 가장 낮은 acc 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 높은 acc의 모델 저장
        if acc > self.highest_val_acc:
            # self.highest_val_acc = acc
            best_model_path = os.path.join(self.result_path+"/"+self.model_name,f'{self.model_name}_Acc_{acc:.4f}_best_model.pt')
            if os.path.exists(self.pre_best):
                os.remove(self.pre_best)
            self.pre_best = best_model_path
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch} epoch result. Acc = {acc:.4f} Loss = {loss:.4f}")

    def load_model(self, model_path: str):
        """모델을 불러오는 함수"""
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")

    def train(self, load_model: bool = False, model_path: str = None) -> None:
        if load_model and model_path:
            self.load_model(model_path)

        wandb.init(project="model_comparison", name=self.model_name)

        val_loss, val_acc = self.validate()
        wandb.log({
            "epoch": 0,
            "train_loss": val_loss,
            "val_accuracy": val_acc
             })
        
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            wandb.log({
            "epoch": epoch+1,
            "train_loss": val_loss,
            "val_accuracy": val_acc
             })
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

            # Loss 및 Accuracy 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # 모델 저장
            self.save_model(epoch=epoch+1, acc=val_acc, loss=val_loss)
            # Early Stopping
            if val_acc > self.highest_val_acc :
                self.highest_val_acc = val_acc
                self.early_stop_counter = 0  
                print(f"Highest Accuracy updated to {self.highest_val_acc:.4f}. Early stop counter reset to 0.")
            else:
                self.early_stop_counter += 1  
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

            self.scheduler.step()

            # 매 에폭마다 그래프 시각화
            #self.plot_results()

        wandb.finish()
        # 최종 그래프 시각화 (전체 학습이 끝난 후에도 그래프를 그릴 수 있음)
        # self.plot_results()  

    def plot_results(self):
        """학습 결과 시각화 (Loss & Accuracy)"""
        epochs_range = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(14, 10))

        # Loss 시각화
        plt.subplot(2, 1, 1)
        plt.plot(epochs_range, self.train_losses, label="Train Loss", color='blue')
        plt.plot(epochs_range, self.val_losses, label="Validation Loss", color='red')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.title("Train & Validation Loss")

        # Accuracy 시각화
        plt.subplot(2, 1, 2)
        plt.plot(epochs_range, self.train_accuracies, label="Train Accuracy", color='green')
        plt.plot(epochs_range, self.val_accuracies, label="Validation Accuracy", color='orange')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.title("Train & Validation Accuracy")

        plt.tight_layout()
        plt.show()


class CLIP_Trainer(BaseTrainer):
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str,
        processor: CLIPProcessor,
        mini_values: list,
        lr: float,
        model_name: str,
        textFrozen: bool = False,
        multi_prompt: bool = False,
        patience: int = 5  # 조기 종료를 위한 patience 설정
        
    ):
        BaseTrainer.__init__(self,
                             model,
                             device,
                             train_loader,
                             val_loader,
                             optimizer,
                             scheduler,
                             loss_fn,
                             epochs,
                             result_path,
                             model_name,
                             patience
                             )
        self.textFrozen = textFrozen
        self.multi_prompt = multi_prompt
        self.processor = processor
        self.mini_values = mini_values
        self.prompt = [line.strip() for line in open(prompt_path, 'r').readlines()]
        if self.multi_prompt:
            self.Val_text = self.generate_prompts(self.mini_values)  
        else:
            self.Val_text = self.mini_values
        self.all_text_features = self.process_text_batches(self.Val_text)

    def generate_prompts(self, labels):     
        prompts = []
        for label in labels:
            class_prompts = [line.replace("{c}",label) for line in self.prompt]
            prompts.extend(class_prompts)
        return prompts
    
    def process_text_batches(self, all_prompts):
        self.model.eval()
        all_text_features = []
        batch_size = self.train_loader.batch_size
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i+batch_size]
            inputs = self.processor(text=batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)# F.normalize(text_features, p=2, dim=1)
                all_text_features.append(text_features)
    
        return torch.cat(all_text_features, dim=0)
    
    def clip_loss(self,similarity: torch.Tensor) -> torch.Tensor:
        image_loss = F.cross_entropy(similarity, torch.arange(similarity.shape[0]).to(similarity.device))
        text_loss = F.cross_entropy(similarity.T, torch.arange(similarity.shape[0]).to(similarity.device))   
        if self.textFrozen:
            return  image_loss
        else:
            return (text_loss + image_loss) / 2.0

    def train_epoch(self) -> (float, float):
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        self.model.to(self.device)
        total_loss = 0.0
        corrects = 0
        total = 0
        if self.textFrozen:
            for param in self.model.text_model.parameters():
                param.requires_grad = False
            
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, targets in progress_bar:

            self.optimizer.zero_grad()
            im = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            if self.multi_prompt:                
                txf = self.process_text_batches(self.generate_prompts(targets))
            else:
                txf = self.process_text_batches(targets)
            imf = self.model.get_image_features(**im)
            imf = imf / imf.norm(dim=1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * imf @ txf.t()
            if self.multi_prompt:
                logits_per_image= logits_per_image.view(logits_per_image.size(0), len(targets), -1).mean(dim=2) 
            loss = self.clip_loss(logits_per_image)     

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            batch_len = len(logits_per_image)
            ground_truth = torch.arange(batch_len).long().to(logits_per_image.device)

            image_preds = torch.argmax(logits_per_image, dim=1)
            image_acc = (image_preds == ground_truth).float().mean().item()

            count = image_acc* batch_len    
            corrects += count
            total += batch_len
            progress_bar.set_postfix(loss=loss.item())


        avg_loss = total_loss / len(self.train_loader)
        accuracy = corrects / total
        return avg_loss, accuracy
    
 

    def validate(self) -> (float, float):
        # 모델의 검증을 진행
        self.model.eval()
        total_loss = 0.0
        corrects = 0
        total = 0

        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)

        all_image_features = []
        all_labels = []
        with torch.no_grad():
            if not self.textFrozen:
                self.all_text_features = self.process_text_batches(self.Val_text)
                
            for images, targets in progress_bar:
                inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

                imf = self.model.get_image_features(**inputs)
                imf = imf / imf.norm(dim=1, keepdim=True)# F.normalize(image_features, p=2, dim=1)
                all_image_features.append(imf)                
                all_labels.append(torch.tensor([self.mini_values.index(ind) for ind in targets],device=self.device))

                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * imf @ self.all_text_features.t()
                if self.multi_prompt:
                    logits_per_image = logits_per_image.view(logits_per_image.size(0), 500, -1).mean(dim=2)               
                   
                _, predicted = logits_per_image.max(1)
                tar = torch.tensor([self.mini_values.index(ind) for ind in targets],device=self.device)
                labels_one_hot = F.one_hot(tar, num_classes=len(self.mini_values)).float().to(self.device)
                loss = self.loss_fn(logits_per_image, labels_one_hot)
                total_loss += loss.item()

                corrects += predicted.eq(tar).sum().item()
                total += len(targets)

                progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(self.val_loader)
        accuracy = corrects / total
        return avg_loss, accuracy
    
 
    
class Trainer(BaseTrainer):
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str,
        model_name: str,
        patience: int = 5  # 조기 종료를 위한 patience 설정
        
    ):
        BaseTrainer.__init__(self,
                             model,
                             device,
                             train_loader,
                             val_loader,
                             optimizer,
                             scheduler,
                             loss_fn,
                             epochs,
                             result_path,
                             model_name,
                             patience
                             )

    
    def train_epoch(self) -> (float, float):
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        total_loss = 0.0
        corrects = 0
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

            _, predicted = outputs.max(1)
            total += targets.size(0)
            corrects += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = corrects / total
        return avg_loss, accuracy

    def validate(self) -> (float, float):
        # 모델의 검증을 진행
        self.model.eval()
        total_loss = 0.0
        corrects = 0
        total = 0

        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                corrects += predicted.eq(targets).sum().item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = corrects / total
        return avg_loss, accuracy
    
  

def get_model_and_transforms(model_name):
   
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT
        model = models.resnet101(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "inception_v3":
        weights=models.Inception_V3_Weights.DEFAULT
        model = models.inception_v3(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT
        model = models.efficientnet_b1(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "vit_l_16":
        weights = models.ViT_L_16_Weights.DEFAULT
        model = models.vit_l_16(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        model = models.convnext_base(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "convnext_large":
        weights = models.ConvNeXt_Large_Weights.DEFAULT
        model = models.convnext_large(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        preprocess = weights.transforms()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model, preprocess

imagenet_cls_map = {'n02119789': 'kit fox, Vulpes macrotis', 'n02100735': 'English setter', 'n02096294': 'Australian terrier', 'n02066245': 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'n02509815': 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 'n02124075': 'Egyptian cat', 'n02417914': 'ibex, Capra ibex', 'n02123394': 'Persian cat', 'n02125311': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'n02423022': 'gazelle', 'n02346627': 'porcupine, hedgehog', 'n02077923': 'sea lion', 'n02447366': 'badger', 'n02109047': 'Great Dane', 'n02092002': 'Scottish deerhound, deerhound', 'n02071294': 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'n02442845': 'mink', 'n02504458': 'African elephant, Loxodonta africana', 'n02114712': 'red wolf, maned wolf, Canis rufus, Canis niger', 'n02128925': 'jaguar, panther, Panthera onca, Felis onca', 'n02117135': 'hyena, hyaena', 'n02493509': 'titi, titi monkey', 'n02457408': 'three-toed sloth, ai, Bradypus tridactylus', 'n02389026': 'sorrel', 'n02443484': 'black-footed ferret, ferret, Mustela nigripes', 'n02110341': 'dalmatian, coach dog, carriage dog', 'n02093256': 'Staffordshire bullterrier, Staffordshire bull terrier', 'n02106382': 'Bouvier des Flandres, Bouviers des Flandres', 'n02441942': 'weasel', 'n02113712': 'miniature poodle', 'n02415577': 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 'n02356798': 'fox squirrel, eastern fox squirrel, Sciurus niger', 'n02488702': 'colobus, colobus monkey', 'n02123159': 'tiger cat', 'n02422699': 'impala, Aepyceros melampus', 'n02114855': 'coyote, prairie wolf, brush wolf, Canis latrans', 'n02094433': 'Yorkshire terrier', 'n02111277': 'Newfoundland, Newfoundland dog', 'n02119022': 'red fox, Vulpes vulpes', 'n02422106': 'hartebeest', 'n02120505': 'grey fox, gray fox, Urocyon cinereoargenteus', 'n02086079': 'Pekinese, Pekingese, Peke', 'n02484975': 'guenon, guenon monkey', 'n02137549': 'mongoose', 'n02500267': 'indri, indris, Indri indri, Indri brevicaudatus', 'n02129604': 'tiger, Panthera tigris', 'n02396427': 'wild boar, boar, Sus scrofa', 'n02391049': 'zebra', 'n02412080': 'ram, tup', 'n02480495': 'orangutan, orang, orangutang, Pongo pygmaeus', 'n02110806': 'basenji', 'n02128385': 'leopard, Panthera pardus', 'n02100583': 'vizsla, Hungarian pointer', 'n02494079': 'squirrel monkey, Saimiri sciureus', 'n02123597': 'Siamese cat, Siamese', 'n02481823': 'chimpanzee, chimp, Pan troglodytes', 'n02105505': 'komondor', 'n02489166': 'proboscis monkey, Nasalis larvatus', 'n02364673': 'guinea pig, Cavia cobaya', 'n02114548': 'white wolf, Arctic wolf, Canis lupus tundrarum', 'n02134084': 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'n02480855': 'gorilla, Gorilla gorilla', 'n02403003': 'ox', 'n02108551': 'Tibetan mastiff', 'n02493793': 'spider monkey, Ateles geoffroyi', 'n02107142': 'Doberman, Doberman pinscher', 'n02397096': 'warthog', 'n02437312': 'Arabian camel, dromedary, Camelus dromedarius', 'n02483708': 'siamang, Hylobates syndactylus, Symphalangus syndactylus', 'n02099601': 'golden retriever', 'n02106166': 'Border collie', 'n02326432': 'hare', 'n02108089': 'boxer', 'n02486261': 'patas, hussar monkey, Erythrocebus patas', 'n02486410': 'baboon', 'n02487347': 'macaque', 'n02492035': 'capuchin, ringtail, Cebus capucinus', 'n02099267': 'flat-coated retriever', 'n02395406': 'hog, pig, grunter, squealer, Sus scrofa', 'n02109961': 'Eskimo dog, husky', 'n02101388': 'Brittany spaniel', 'n03187595': 'dial telephone, dial phone', 'n03733281': 'maze, labyrinth', 'n02101006': 'Gordon setter', 'n02115641': 'dingo, warrigal, warragal, Canis dingo', 'n02342885': 'hamster', 'n02120079': 'Arctic fox, white fox, Alopex lagopus', 'n02408429': 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis', 'n02133161': 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'n02328150': 'Angora, Angora rabbit', 'n02410509': 'bison', 'n02492660': 'howler monkey, howler', 'n02398521': 'hippopotamus, hippo, river horse, Hippopotamus amphibius', 'n02510455': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 'n02123045': 'tabby, tabby cat', 'n02490219': 'marmoset', 'n02109525': 'Saint Bernard, St Bernard', 'n02454379': 'armadillo', 'n02090379': 'redbone', 'n02443114': 'polecat, fitch, foulmart, foumart, Mustela putorius', 'n02361337': 'marmot', 'n02483362': 'gibbon, Hylobates lar', 'n02437616': 'llama', 'n02325366': 'wood rabbit, cottontail, cottontail rabbit', 'n02129165': 'lion, king of beasts, Panthera leo', 'n02100877': 'Irish setter, red setter', 'n02074367': 'dugong, Dugong dugon', 'n02504013': 'Indian elephant, Elephas maximus', 'n02363005': 'beaver', 'n02497673': 'Madagascar cat, ring-tailed lemur, Lemur catta', 'n02087394': 'Rhodesian ridgeback', 'n02127052': 'lynx, catamount', 'n02116738': 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'n02488291': 'langur', 'n02114367': 'timber wolf, grey wolf, gray wolf, Canis lupus', 'n02130308': 'cheetah, chetah, Acinonyx jubatus', 'n02134418': 'sloth bear, Melursus ursinus, Ursus ursinus', 'n02106662': 'German shepherd, German shepherd dog, German police dog, alsatian', 'n02444819': 'otter', 'n01882714': 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'n01871265': 'tusker', 'n01872401': 'echidna, spiny anteater, anteater', 'n01877812': 'wallaby, brush kangaroo', 'n01873310': 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus', 'n01883070': 'wombat', 'n04086273': 'revolver, six-gun, six-shooter', 'n04507155': 'umbrella', 'n04147183': 'schooner', 'n04254680': 'soccer ball', 'n02672831': 'accordion, piano accordion, squeeze box', 'n02219486': 'ant, emmet, pismire', 'n02317335': 'starfish, sea star', 'n01968897': 'chambered nautilus, pearly nautilus, nautilus', 'n03452741': 'grand piano, grand', 'n03642806': 'laptop, laptop computer', 'n07745940': 'strawberry', 'n02690373': 'airliner', 'n04552348': 'warplane, military plane', 'n02692877': 'airship, dirigible', 'n02782093': 'balloon', 'n04266014': 'space shuttle', 'n03344393': 'fireboat', 'n03447447': 'gondola', 'n04273569': 'speedboat', 'n03662601': 'lifeboat', 'n02951358': 'canoe', 'n04612504': 'yawl', 'n02981792': 'catamaran', 'n04483307': 'trimaran', 'n03095699': 'container ship, containership, container vessel', 'n03673027': 'liner, ocean liner', 'n03947888': 'pirate, pirate ship', 'n02687172': 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'n04347754': 'submarine, pigboat, sub, U-boat', 'n04606251': 'wreck', 'n03478589': 'half track', 'n04389033': 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'n03773504': 'missile', 'n02860847': 'bobsled, bobsleigh, bob', 'n03218198': 'dogsled, dog sled, dog sleigh', 'n02835271': 'bicycle-built-for-two, tandem bicycle, tandem', 'n03792782': 'mountain bike, all-terrain bike, off-roader', 'n03393912': 'freight car', 'n03895866': 'passenger car, coach, carriage', 'n02797295': 'barrow, garden cart, lawn cart, wheelbarrow', 'n04204347': 'shopping cart', 'n03791053': 'motor scooter, scooter', 'n03384352': 'forklift', 'n03272562': 'electric locomotive', 'n04310018': 'steam locomotive', 'n02704792': 'amphibian, amphibious vehicle', 'n02701002': 'ambulance', 'n02814533': 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'n02930766': 'cab, hack, taxi, taxicab', 'n03100240': 'convertible', 'n03594945': 'jeep, landrover', 'n03670208': 'limousine, limo', 'n03770679': 'minivan', 'n03777568': 'Model T', 'n04037443': 'racer, race car, racing car', 'n04285008': 'sports car, sport car', 'n03444034': 'go-kart', 'n03445924': 'golfcart, golf cart', 'n03785016': 'moped', 'n04252225': 'snowplow, snowplough', 'n03345487': 'fire engine, fire truck', 'n03417042': 'garbage truck, dustcart', 'n03930630': 'pickup, pickup truck', 'n04461696': 'tow truck, tow car, wrecker', 'n04467665': 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'n03796401': 'moving van', 'n03977966': 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'n04065272': 'recreational vehicle, RV, R.V.', 'n04335435': 'streetcar, tram, tramcar, trolley, trolley car', 'n04252077': 'snowmobile', 'n04465501': 'tractor', 'n03776460': 'mobile home, manufactured home', 'n04482393': 'tricycle, trike, velocipede', 'n04509417': 'unicycle, monocycle', 'n03538406': 'horse cart, horse-cart', 'n03788365': 'mosquito net', 'n03868242': 'oxcart', 'n02804414': 'bassinet', 'n03125729': 'cradle', 'n03131574': 'crib, cot', 'n03388549': 'four-poster', 'n02870880': 'bookcase', 'n03018349': 'china cabinet, china closet', 'n03742115': 'medicine chest, medicine cabinet', 'n03016953': 'chiffonier, commode', 'n04380533': 'table lamp', 'n03337140': 'file, file cabinet, filing cabinet', 'n03902125': 'pay-phone, pay-station', 'n03891251': 'park bench', 'n02791124': 'barber chair', 'n04429376': 'throne', 'n03376595': 'folding chair', 'n04099969': 'rocking chair, rocker', 'n04344873': 'studio couch, day bed', 'n04447861': 'toilet seat', 'n03179701': 'desk', 'n03982430': 'pool table, billiard table, snooker table', 'n03201208': 'dining table, board', 'n03290653': 'entertainment center', 'n04550184': 'wardrobe, closet, press', 'n07742313': 'Granny Smith', 'n07747607': 'orange', 'n07749582': 'lemon', 'n07753113': 'fig', 'n07753275': 'pineapple, ananas', 'n07753592': 'banana', 'n07754684': 'jackfruit, jak, jack', 'n07760859': 'custard apple', 'n07768694': 'pomegranate', 'n12267677': 'acorn', 'n12620546': 'hip, rose hip, rosehip', 'n13133613': 'ear, spike, capitulum', 'n11879895': 'rapeseed', 'n12144580': 'corn', 'n12768682': 'buckeye, horse chestnut, conker', 'n03854065': 'organ, pipe organ', 'n04515003': 'upright, upright piano', 'n03017168': 'chime, bell, gong', 'n03249569': 'drum, membranophone, tympan', 'n03447721': 'gong, tam-tam', 'n03720891': 'maraca', 'n03721384': 'marimba, xylophone', 'n04311174': 'steel drum', 'n02787622': 'banjo', 'n02992211': 'cello, violoncello', 'n03637318': 'lampshade, lamp shade', 'n03495258': 'harp', 'n02676566': 'acoustic guitar', 'n03272010': 'electric guitar', 'n03110669': 'cornet, horn, trumpet, trump', 'n03394916': 'French horn, horn', 'n04487394': 'trombone', 'n03494278': 'harmonica, mouth organ, harp, mouth harp', 'n03840681': 'ocarina, sweet potato', 'n03884397': 'panpipe, pandean pipe, syrinx', 'n02804610': 'bassoon', 'n04141076': 'sax, saxophone', 'n03372029': 'flute, transverse flute', 'n11939491': 'daisy', 'n12057211': "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'n09246464': 'cliff, drop, drop-off', 'n09468604': 'valley, vale', 'n09193705': 'alp', 'n09472597': 'volcano', 'n09399592': 'promontory, headland, head, foreland', 'n09421951': 'sandbar, sand bar', 'n09256479': 'coral reef', 'n09332890': 'lakeside, lakeshore', 'n09428293': 'seashore, coast, seacoast, sea-coast', 'n09288635': 'geyser', 'n03498962': 'hatchet', 'n03041632': 'cleaver, meat cleaver, chopper', 'n03658185': 'letter opener, paper knife, paperknife', 'n03954731': "plane, carpenter's plane, woodworking plane", 'n03995372': 'power drill', 'n03649909': 'lawn mower, mower', 'n03481172': 'hammer', 'n03109150': 'corkscrew, bottle screw', 'n02951585': 'can opener, tin opener', 'n03970156': "plunger, plumber's helper", 'n04154565': 'screwdriver', 'n04208210': 'shovel', 'n03967562': 'plow, plough', 'n03000684': 'chain saw, chainsaw', 'n01514668': 'cock', 'n01514859': 'hen', 'n01518878': 'ostrich, Struthio camelus', 'n01530575': 'brambling, Fringilla montifringilla', 'n01531178': 'goldfinch, Carduelis carduelis', 'n01532829': 'house finch, linnet, Carpodacus mexicanus', 'n01534433': 'junco, snowbird', 'n01537544': 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'n01558993': 'robin, American robin, Turdus migratorius', 'n01560419': 'bulbul', 'n01580077': 'jay', 'n01582220': 'magpie', 'n01592084': 'chickadee', 'n01601694': 'water ouzel, dipper', 'n01608432': 'kite', 'n01614925': 'bald eagle, American eagle, Haliaeetus leucocephalus', 'n01616318': 'vulture', 'n01622779': 'great grey owl, great gray owl, Strix nebulosa', 'n01795545': 'black grouse', 'n01796340': 'ptarmigan', 'n01797886': 'ruffed grouse, partridge, Bonasa umbellus', 'n01798484': 'prairie chicken, prairie grouse, prairie fowl', 'n01806143': 'peacock', 'n01806567': 'quail', 'n01807496': 'partridge', 'n01817953': 'African grey, African gray, Psittacus erithacus', 'n01818515': 'macaw', 'n01819313': 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'n01820546': 'lorikeet', 'n01824575': 'coucal', 'n01828970': 'bee eater', 'n01829413': 'hornbill', 'n01833805': 'hummingbird', 'n01843065': 'jacamar', 'n01843383': 'toucan', 'n01847000': 'drake', 'n01855032': 'red-breasted merganser, Mergus serrator', 'n01855672': 'goose', 'n01860187': 'black swan, Cygnus atratus', 'n02002556': 'white stork, Ciconia ciconia', 'n02002724': 'black stork, Ciconia nigra', 'n02006656': 'spoonbill', 'n02007558': 'flamingo', 'n02009912': 'American egret, great white heron, Egretta albus', 'n02009229': 'little blue heron, Egretta caerulea', 'n02011460': 'bittern', 'n02012849': 'crane', 'n02013706': 'limpkin, Aramus pictus', 'n02018207': 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'n02018795': 'bustard', 'n02025239': 'ruddy turnstone, Arenaria interpres', 'n02027492': 'red-backed sandpiper, dunlin, Erolia alpina', 'n02028035': 'redshank, Tringa totanus', 'n02033041': 'dowitcher', 'n02037110': 'oystercatcher, oyster catcher', 'n02017213': 'European gallinule, Porphyrio porphyrio', 'n02051845': 'pelican', 'n02056570': 'king penguin, Aptenodytes patagonica', 'n02058221': 'albatross, mollymawk', 'n01484850': 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'n01491361': 'tiger shark, Galeocerdo cuvieri', 'n01494475': 'hammerhead, hammerhead shark', 'n01496331': 'electric ray, crampfish, numbfish, torpedo', 'n01498041': 'stingray', 'n02514041': 'barracouta, snoek', 'n02536864': 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch', 'n01440764': 'tench, Tinca tinca', 'n01443537': 'goldfish, Carassius auratus', 'n02526121': 'eel', 'n02606052': 'rock beauty, Holocanthus tricolor', 'n02607072': 'anemone fish', 'n02643566': 'lionfish', 'n02655020': 'puffer, pufferfish, blowfish, globefish', 'n02640242': 'sturgeon', 'n02641379': 'gar, garfish, garpike, billfish, Lepisosteus osseus', 'n01664065': 'loggerhead, loggerhead turtle, Caretta caretta', 'n01667114': 'mud turtle', 'n01667778': 'terrapin', 'n01669191': 'box turtle, box tortoise', 'n01675722': 'banded gecko', 'n01677366': 'common iguana, iguana, Iguana iguana', 'n01682714': 'American chameleon, anole, Anolis carolinensis', 'n01685808': 'whiptail, whiptail lizard', 'n01687978': 'agama', 'n01688243': 'frilled lizard, Chlamydosaurus kingi', 'n01689811': 'alligator lizard', 'n01692333': 'Gila monster, Heloderma suspectum', 'n01693334': 'green lizard, Lacerta viridis', 'n01694178': 'African chameleon, Chamaeleo chamaeleon', 'n01695060': 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 'n01704323': 'triceratops', 'n01697457': 'African crocodile, Nile crocodile, Crocodylus niloticus', 'n01698640': 'American alligator, Alligator mississipiensis', 'n01728572': 'thunder snake, worm snake, Carphophis amoenus', 'n01728920': 'ringneck snake, ring-necked snake, ring snake', 'n01729322': 'hognose snake, puff adder, sand viper', 'n01729977': 'green snake, grass snake', 'n01734418': 'king snake, kingsnake', 'n01735189': 'garter snake, grass snake', 'n01737021': 'water snake', 'n01739381': 'vine snake', 'n01740131': 'night snake, Hypsiglena torquata', 'n01742172': 'boa constrictor, Constrictor constrictor', 'n01744401': 'rock python, rock snake, Python sebae', 'n01748264': 'Indian cobra, Naja naja', 'n01749939': 'green mamba', 'n01751748': 'sea snake', 'n01753488': 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 'n04326547': 'stone wall', 'n01756291': 'sidewinder, horned rattlesnake, Crotalus cerastes', 'n01629819': 'European fire salamander, Salamandra salamandra', 'n01630670': 'common newt, Triturus vulgaris', 'n01631663': 'eft', 'n01632458': 'spotted salamander, Ambystoma maculatum', 'n01632777': 'axolotl, mud puppy, Ambystoma mexicanum', 'n01641577': 'bullfrog, Rana catesbeiana', 'n01644373': 'tree frog, tree-frog', 'n01644900': 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 'n04579432': 'whistle', 'n04592741': 'wing', 'n03876231': 'paintbrush', 'n03868863': 'oxygen mask', 'n04251144': 'snorkel', 'n03691459': 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', 'n03759954': 'microphone, mike', 'n04152593': 'screen, CRT screen', 'n03793489': 'mouse, computer mouse', 'n03271574': 'electric fan, blower', 'n03843555': 'oil filter', 'n04332243': 'strainer', 'n04265275': 'space heater', 'n04330267': 'stove', 'n03467068': 'guillotine', 'n02794156': 'barometer', 'n04118776': 'rule, ruler', 'n03841143': 'odometer, hodometer, mileometer, milometer', 'n04141975': 'scale, weighing machine', 'n02708093': 'analog clock', 'n03196217': 'digital clock', 'n04548280': 'wall clock', 'n03544143': 'hourglass', 'n04355338': 'sundial', 'n03891332': 'parking meter', 'n04328186': 'stopwatch, stop watch', 'n03197337': 'digital watch', 'n04317175': 'stethoscope', 'n04376876': 'syringe', 'n03706229': 'magnetic compass', 'n02841315': 'binoculars, field glasses, opera glasses', 'n04009552': 'projector', 'n04356056': 'sunglasses, dark glasses, shades', 'n03692522': "loupe, jeweler's loupe", 'n04044716': 'radio telescope, radio reflector', 'n02879718': 'bow', 'n02950826': 'cannon', 'n02749479': 'assault rifle, assault gun', 'n04090263': 'rifle', 'n04008634': 'projectile, missile', 'n03085013': 'computer keyboard, keypad', 'n04505470': 'typewriter keyboard', 'n03126707': 'crane', 'n03666591': 'lighter, light, igniter, ignitor', 'n02666196': 'abacus', 'n02977058': 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'n04238763': 'slide rule, slipstick', 'n03180011': 'desktop computer', 'n03485407': 'hand-held computer, hand-held microcomputer', 'n03832673': 'notebook, notebook computer', 'n03874599': 'padlock', 'n03496892': 'harvester, reaper', 'n04428191': 'thresher, thrasher, threshing machine', 'n04004767': 'printer', 'n04243546': 'slot, one-armed bandit', 'n04525305': 'vending machine', 'n04179913': 'sewing machine', 'n03602883': 'joystick', 'n04372370': 'switch, electric switch, electrical switch', 'n03532672': 'hook, claw', 'n02974003': 'car wheel', 'n03874293': 'paddlewheel, paddle wheel', 'n03944341': 'pinwheel', 'n03992509': "potter's wheel", 'n03425413': 'gas pump, gasoline pump, petrol pump, island dispenser', 'n02966193': 'carousel, carrousel, merry-go-round, roundabout, whirligig', 'n04371774': 'swing', 'n04067472': 'reel', 'n04040759': 'radiator', 'n04019541': 'puck, hockey puck', 'n03492542': 'hard disc, hard disk, fixed disk', 'n04355933': 'sunglass', 'n03929660': 'pick, plectrum, plectron', 'n02965783': 'car mirror', 'n04258138': 'solar dish, solar collector, solar furnace', 'n04074963': 'remote control, remote', 'n03208938': 'disk brake, disc brake', 'n02910353': 'buckle', 'n03476684': 'hair slide', 'n03627232': 'knot', 'n03075370': 'combination lock', 'n06359193': 'web site, website, internet site, site', 'n03804744': 'nail', 'n04127249': 'safety pin', 'n04153751': 'screw', 'n03803284': 'muzzle', 'n04162706': 'seat belt, seatbelt', 'n04228054': 'ski', 'n02948072': 'candle, taper, wax light', 'n03590841': "jack-o'-lantern", 'n04286575': 'spotlight, spot', 'n04456115': 'torch', 'n03814639': 'neck brace', 'n03933933': 'pier', 'n04485082': 'tripod', 'n03733131': 'maypole', 'n03483316': 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'n03794056': 'mousetrap', 'n04275548': "spider web, spider's web", 'n01768244': 'trilobite', 'n01770081': 'harvestman, daddy longlegs, Phalangium opilio', 'n01770393': 'scorpion', 'n01773157': 'black and gold garden spider, Argiope aurantia', 'n01773549': 'barn spider, Araneus cavaticus', 'n01773797': 'garden spider, Aranea diademata', 'n01774384': 'black widow, Latrodectus mactans', 'n01774750': 'tarantula', 'n01775062': 'wolf spider, hunting spider', 'n01776313': 'tick', 'n01784675': 'centipede', 'n01990800': 'isopod', 'n01978287': 'Dungeness crab, Cancer magister', 'n01978455': 'rock crab, Cancer irroratus', 'n01980166': 'fiddler crab', 'n01981276': 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 'n01983481': 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'n01984695': 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'n01985128': 'crayfish, crawfish, crawdad, crawdaddy', 'n01986214': 'hermit crab', 'n02165105': 'tiger beetle', 'n02165456': 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'n02167151': 'ground beetle, carabid beetle', 'n02168699': 'long-horned beetle, longicorn, longicorn beetle', 'n02169497': 'leaf beetle, chrysomelid', 'n02172182': 'dung beetle', 'n02174001': 'rhinoceros beetle', 'n02177972': 'weevil', 'n02190166': 'fly', 'n02206856': 'bee', 'n02226429': 'grasshopper, hopper', 'n02229544': 'cricket', 'n02231487': 'walking stick, walkingstick, stick insect', 'n02233338': 'cockroach, roach', 'n02236044': 'mantis, mantid', 'n02256656': 'cicada, cicala', 'n02259212': 'leafhopper', 'n02264363': 'lacewing, lacewing fly', 'n02268443': "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'n02268853': 'damselfly', 'n02276258': 'admiral', 'n02277742': 'ringlet, ringlet butterfly', 'n02279972': 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'n02280649': 'cabbage butterfly', 'n02281406': 'sulphur butterfly, sulfur butterfly', 'n02281787': 'lycaenid, lycaenid butterfly', 'n01910747': 'jellyfish', 'n01914609': 'sea anemone, anemone', 'n01917289': 'brain coral', 'n01924916': 'flatworm, platyhelminth', 'n01930112': 'nematode, nematode worm, roundworm', 'n01943899': 'conch', 'n01944390': 'snail', 'n01945685': 'slug', 'n01950731': 'sea slug, nudibranch', 'n01955084': 'chiton, coat-of-mail shell, sea cradle, polyplacophore', 'n02319095': 'sea urchin', 'n02321529': 'sea cucumber, holothurian', 'n03584829': 'iron, smoothing iron', 'n03297495': 'espresso maker', 'n03761084': 'microwave, microwave oven', 'n03259280': 'Dutch oven', 'n04111531': 'rotisserie', 'n04442312': 'toaster', 'n04542943': 'waffle iron', 'n04517823': 'vacuum, vacuum cleaner', 'n03207941': 'dishwasher, dish washer, dishwashing machine', 'n04070727': 'refrigerator, icebox', 'n04554684': 'washer, automatic washer, washing machine', 'n03133878': 'Crock Pot', 'n03400231': 'frying pan, frypan, skillet', 'n04596742': 'wok', 'n02939185': 'caldron, cauldron', 'n03063689': 'coffeepot', 'n04398044': 'teapot', 'n04270147': 'spatula', 'n02699494': 'altar', 'n04486054': 'triumphal arch', 'n03899768': 'patio, terrace', 'n04311004': 'steel arch bridge', 'n04366367': 'suspension bridge', 'n04532670': 'viaduct', 'n02793495': 'barn', 'n03457902': 'greenhouse, nursery, glasshouse', 'n03877845': 'palace', 'n03781244': 'monastery', 'n03661043': 'library', 'n02727426': 'apiary, bee house', 'n02859443': 'boathouse', 'n03028079': 'church, church building', 'n03788195': 'mosque', 'n04346328': 'stupa, tope', 'n03956157': 'planetarium', 'n04081281': 'restaurant, eating house, eating place, eatery', 'n03032252': 'cinema, movie theater, movie theatre, movie house, picture palace', 'n03529860': 'home theater, home theatre', 'n03697007': 'lumbermill, sawmill', 'n03065424': 'coil, spiral, volute, whorl, helix', 'n03837869': 'obelisk', 'n04458633': 'totem pole', 'n02980441': 'castle', 'n04005630': 'prison, prison house', 'n03461385': 'grocery store, grocery, food market, market', 'n02776631': 'bakery, bakeshop, bakehouse', 'n02791270': 'barbershop', 'n02871525': 'bookshop, bookstore, bookstall', 'n02927161': 'butcher shop, meat market', 'n03089624': 'confectionery, confectionary, candy store', 'n04200800': 'shoe shop, shoe-shop, shoe store', 'n04443257': 'tobacco shop, tobacconist shop, tobacconist', 'n04462240': 'toyshop', 'n03388043': 'fountain', 'n03042490': 'cliff dwelling', 'n04613696': 'yurt', 'n03216828': 'dock, dockage, docking facility', 'n02892201': 'brass, memorial tablet, plaque', 'n03743016': 'megalith, megalithic structure', 'n02788148': 'bannister, banister, balustrade, balusters, handrail', 'n02894605': 'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 'n03160309': 'dam, dike, dyke', 'n03000134': 'chainlink fence', 'n03930313': 'picket fence, paling', 'n04604644': 'worm fence, snake fence, snake-rail fence, Virginia fence', 'n01755581': 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'n03459775': 'grille, radiator grille', 'n04239074': 'sliding door', 'n04501370': 'turnstile', 'n03792972': 'mountain tent', 'n04149813': 'scoreboard', 'n03530642': 'honeycomb', 'n03961711': 'plate rack', 'n03903868': 'pedestal, plinth, footstall', 'n02814860': 'beacon, lighthouse, beacon light, pharos', 'n01665541': 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'n07711569': 'mashed potato', 'n07720875': 'bell pepper', 'n07714571': 'head cabbage', 'n07714990': 'broccoli', 'n07715103': 'cauliflower', 'n07716358': 'zucchini, courgette', 'n07716906': 'spaghetti squash', 'n07717410': 'acorn squash', 'n07717556': 'butternut squash', 'n07718472': 'cucumber, cuke', 'n07718747': 'artichoke, globe artichoke', 'n07730033': 'cardoon', 'n07734744': 'mushroom', 'n04209239': 'shower curtain', 'n03594734': 'jean, blue jean, denim', 'n02971356': 'carton', 'n03485794': 'handkerchief, hankie, hanky, hankey', 'n04133789': 'sandal', 'n02747177': 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'n04125021': 'safe', 'n07579787': 'plate', 'n03814906': 'necklace', 'n03134739': 'croquet ball', 'n03404251': 'fur coat', 'n04423845': 'thimble', 'n03877472': "pajama, pyjama, pj's, jammies", 'n04120489': 'running shoe', 'n03838899': 'oboe, hautboy, hautbois', 'n03062245': 'cocktail shaker', 'n03014705': 'chest', 'n03717622': 'manhole cover', 'n03777754': 'modem', 'n04493381': 'tub, vat', 'n04476259': 'tray', 'n02777292': 'balance beam, beam', 'n07693725': 'bagel, beigel', 'n04536866': 'violin, fiddle', 'n03998194': 'prayer rug, prayer mat', 'n03617480': 'kimono', 'n07590611': 'hot pot, hotpot', 'n04579145': 'whiskey jug', 'n03623198': 'knee pad', 'n07248320': 'book jacket, dust cover, dust jacket, dust wrapper', 'n04277352': 'spindle', 'n04229816': 'ski mask', 'n02823428': 'beer bottle', 'n03127747': 'crash helmet', 'n02877765': 'bottlecap', 'n04435653': 'tile roof', 'n03724870': 'mask', 'n03710637': 'maillot', 'n03920288': 'Petri dish', 'n03379051': 'football helmet', 'n02807133': 'bathing cap, swimming cap', 'n04399382': 'teddy, teddy bear', 'n03527444': 'holster', 'n03983396': 'pop bottle, soda bottle', 'n03924679': 'photocopier', 'n04532106': 'vestment', 'n06785654': 'crossword puzzle, crossword', 'n03445777': 'golf ball', 'n07613480': 'trifle', 'n04350905': 'suit, suit of clothes', 'n04562935': 'water tower', 'n03325584': 'feather boa, boa', 'n03045698': 'cloak', 'n07892512': 'red wine', 'n03250847': 'drumstick', 'n04192698': 'shield, buckler', 'n03026506': 'Christmas stocking', 'n03534580': 'hoopskirt, crinoline', 'n07565083': 'menu', 'n04296562': 'stage', 'n02869837': 'bonnet, poke bonnet', 'n07871810': 'meat loaf, meatloaf', 'n02799071': 'baseball', 'n03314780': 'face powder', 'n04141327': 'scabbard', 'n04357314': 'sunscreen, sunblock, sun blocker', 'n02823750': 'beer glass', 'n13052670': 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa', 'n07583066': 'guacamole', 'n04599235': 'wool, woolen, woollen', 'n07802026': 'hay', 'n02883205': 'bow tie, bow-tie, bowtie', 'n03709823': 'mailbag, postbag', 'n04560804': 'water jug', 'n02909870': 'bucket, pail', 'n03207743': 'dishrag, dishcloth', 'n04263257': 'soup bowl', 'n07932039': 'eggnog', 'n03786901': 'mortar', 'n04479046': 'trench coat', 'n03873416': 'paddle, boat paddle', 'n02999410': 'chain', 'n04367480': 'swab, swob, mop', 'n03775546': 'mixing bowl', 'n07875152': 'potpie', 'n04591713': 'wine bottle', 'n04201297': 'shoji', 'n02916936': 'bulletproof vest', 'n03240683': 'drilling platform, offshore rig', 'n02840245': 'binder, ring-binder', 'n02963159': 'cardigan', 'n04370456': 'sweatshirt', 'n03991062': 'pot, flowerpot', 'n02843684': 'birdhouse', 'n03599486': 'jinrikisha, ricksha, rickshaw', 'n03482405': 'hamper', 'n03942813': 'ping-pong ball', 'n03908618': 'pencil box, pencil case', 'n07584110': 'consomme', 'n02730930': 'apron', 'n04023962': 'punching bag, punch bag, punching ball, punchball', 'n02769748': 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'n10148035': 'groom, bridegroom', 'n02817516': 'bearskin, busby, shako', 'n03908714': 'pencil sharpener', 'n02906734': 'broom', 'n02667093': 'abaya', 'n03787032': 'mortarboard', 'n03980874': 'poncho', 'n03141823': 'crutch', 'n03976467': 'Polaroid camera, Polaroid Land camera', 'n04264628': 'space bar', 'n07930864': 'cup', 'n04039381': 'racket, racquet', 'n06874185': 'traffic light, traffic signal, stoplight', 'n04033901': 'quill, quill pen', 'n04041544': 'radio, wireless', 'n02128757': 'snow leopard, ounce, Panthera uncia', 'n07860988': 'dough', 'n03146219': 'cuirass', 'n03763968': 'military uniform', 'n03676483': 'lipstick, lip rouge', 'n04209133': 'shower cap', 'n03782006': 'monitor', 'n03857828': 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'n03775071': 'mitten', 'n02892767': 'brassiere, bra, bandeau', 'n07684084': 'French loaf', 'n04522168': 'vase', 'n03764736': 'milk can', 'n04118538': 'rugby ball', 'n03887697': 'paper towel', 'n13044778': 'earthstar', 'n03291819': 'envelope', 'n03770439': 'miniskirt, mini', 'n03124170': 'cowboy hat, ten-gallon hat', 'n04487081': 'trolleybus, trolley coach, trackless trolley', 'n03916031': 'perfume, essence', 'n02808440': 'bathtub, bathing tub, bath, tub', 'n07697537': 'hotdog, hot dog, red hot', 'n12985857': 'coral fungus', 'n02917067': 'bullet train, bullet', 'n03938244': 'pillow', 'n15075141': 'toilet tissue, toilet paper, bathroom tissue', 'n02978881': 'cassette', 'n02966687': "carpenter's kit, tool kit", 'n03633091': 'ladle', 'n13040303': 'stinkhorn, carrion fungus', 'n03690938': 'lotion', 'n03476991': 'hair spray', 'n02669723': "academic gown, academic robe, judge's robe", 'n03220513': 'dome', 'n03127925': 'crate', 'n04584207': 'wig', 'n07880968': 'burrito', 'n03937543': 'pill bottle', 'n03000247': 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour', 'n04418357': 'theater curtain, theatre curtain', 'n04590129': 'window shade', 'n02795169': 'barrel, cask', 'n04553703': 'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'n02783161': 'ballpoint, ballpoint pen, ballpen, Biro', 'n02802426': 'basketball', 'n02808304': 'bath towel', 'n03124043': 'cowboy boot', 'n03450230': 'gown', 'n04589890': 'window screen', 'n12998815': 'agaric', 'n02113799': 'standard poodle', 'n02992529': 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'n03825788': 'nipple', 'n02790996': 'barbell', 'n03710193': 'mailbox, letter box', 'n03630383': 'lab coat, laboratory coat', 'n03347037': 'fire screen, fireguard', 'n03769881': 'minibus', 'n03871628': 'packet', 'n02132136': 'brown bear, bruin, Ursus arctos', 'n03976657': 'pole', 'n03535780': 'horizontal bar, high bar', 'n04259630': 'sombrero', 'n03929855': 'pickelhaube', 'n04049303': 'rain barrel', 'n04548362': 'wallet, billfold, notecase, pocketbook', 'n02979186': 'cassette player', 'n06596364': 'comic book', 'n03935335': 'piggy bank, penny bank', 'n06794110': 'street sign', 'n02825657': 'bell cote, bell cot', 'n03388183': 'fountain pen', 'n04591157': 'Windsor tie', 'n04540053': 'volleyball', 'n03866082': 'overskirt', 'n04136333': 'sarong', 'n04026417': 'purse', 'n02865351': 'bolo tie, bolo, bola tie, bola', 'n02834397': 'bib', 'n03888257': 'parachute, chute', 'n04235860': 'sleeping bag', 'n04404412': 'television, television system', 'n04371430': 'swimming trunks, bathing trunks', 'n03733805': 'measuring cup', 'n07920052': 'espresso', 'n07873807': 'pizza, pizza pie', 'n02895154': 'breastplate, aegis, egis', 'n04204238': 'shopping basket', 'n04597913': 'wooden spoon', 'n04131690': 'saltshaker, salt shaker', 'n07836838': 'chocolate sauce, chocolate syrup', 'n09835506': 'ballplayer, baseball player', 'n03443371': 'goblet', 'n13037406': 'gyromitra', 'n04336792': 'stretcher', 'n04557648': 'water bottle', 'n02445715': 'skunk, polecat, wood pussy', 'n04254120': 'soap dispenser', 'n03595614': 'jersey, T-shirt, tee shirt', 'n04146614': 'school bus', 'n03598930': 'jigsaw puzzle', 'n03958227': 'plastic bag', 'n04069434': 'reflex camera', 'n03188531': 'diaper, nappy, napkin', 'n02786058': 'Band Aid', 'n07615774': 'ice lolly, lolly, lollipop, popsicle', 'n04525038': 'velvet', 'n04409515': 'tennis ball', 'n03424325': 'gasmask, respirator, gas helmet', 'n03223299': 'doormat, welcome mat', 'n03680355': 'Loafer', 'n07614500': 'ice cream, icecream', 'n07695742': 'pretzel', 'n04033995': 'quilt, comforter, comfort, puff', 'n03710721': 'maillot, tank suit', 'n04392985': 'tape player', 'n03047690': 'clog, geta, patten, sabot', 'n03584254': 'iPod', 'n13054560': 'bolete', 'n02138441': 'meerkat, mierkat', 'n10565667': 'scuba diver', 'n03950228': 'pitcher, ewer', 'n03729826': 'matchstick', 'n02837789': 'bikini, two-piece', 'n04254777': 'sock', 'n02988304': 'CD player', 'n03657121': 'lens cap, lens cover', 'n04417672': 'thatch, thatched roof', 'n04523525': 'vault', 'n02815834': 'beaker', 'n09229709': 'bubble', 'n07697313': 'cheeseburger', 'n03888605': 'parallel bars, bars', 'n03355925': 'flagpole, flagstaff', 'n03063599': 'coffee mug', 'n04116512': 'rubber eraser, rubber, pencil eraser', 'n04325704': 'stole', 'n07831146': 'carbonara', 'n03255030': 'dumbbell', 'n02110185': 'Siberian husky', 'n02102040': 'English springer, English springer spaniel', 'n02110063': 'malamute, malemute, Alaskan malamute', 'n02089867': 'Walker hound, Walker foxhound', 'n02102177': 'Welsh springer spaniel', 'n02091134': 'whippet', 'n02092339': 'Weimaraner', 'n02098105': 'soft-coated wheaten terrier', 'n02096437': 'Dandie Dinmont, Dandie Dinmont terrier', 'n02105641': 'Old English sheepdog, bobtail', 'n02091635': 'otterhound, otter hound', 'n02088466': 'bloodhound, sleuthhound', 'n02096051': 'Airedale, Airedale terrier', 'n02097130': 'giant schnauzer', 'n02089078': 'black-and-tan coonhound', 'n02086910': 'papillon', 'n02113978': 'Mexican hairless', 'n02113186': 'Cardigan, Cardigan Welsh corgi', 'n02105162': 'malinois', 'n02098413': 'Lhasa, Lhasa apso', 'n02091467': 'Norwegian elkhound, elkhound', 'n02106550': 'Rottweiler', 'n02091831': 'Saluki, gazelle hound', 'n02104365': 'schipperke', 'n02112706': 'Brabancon griffon', 'n02098286': 'West Highland white terrier', 'n02095889': 'Sealyham terrier, Sealyham', 'n02090721': 'Irish wolfhound', 'n02108000': 'EntleBucher', 'n02108915': 'French bulldog', 'n02107683': 'Bernese mountain dog', 'n02085936': 'Maltese dog, Maltese terrier, Maltese', 'n02094114': 'Norfolk terrier', 'n02087046': 'toy terrier', 'n02096177': 'cairn, cairn terrier', 'n02105056': 'groenendael', 'n02101556': 'clumber, clumber spaniel', 'n02088094': 'Afghan hound, Afghan', 'n02085782': 'Japanese spaniel', 'n02090622': 'borzoi, Russian wolfhound', 'n02113624': 'toy poodle', 'n02093859': 'Kerry blue terrier', 'n02097298': 'Scotch terrier, Scottish terrier, Scottie', 'n02096585': 'Boston bull, Boston terrier', 'n02107574': 'Greater Swiss Mountain dog', 'n02107908': 'Appenzeller', 'n02086240': 'Shih-Tzu', 'n02102973': 'Irish water spaniel', 'n02112018': 'Pomeranian', 'n02093647': 'Bedlington terrier', 'n02097047': 'miniature schnauzer', 'n02106030': 'collie', 'n02093991': 'Irish terrier', 'n02110627': 'affenpinscher, monkey pinscher, monkey dog', 'n02097658': 'silky terrier, Sydney silky', 'n02088364': 'beagle', 'n02111129': 'Leonberg', 'n02100236': 'German short-haired pointer', 'n02115913': 'dhole, Cuon alpinus', 'n02099849': 'Chesapeake Bay retriever', 'n02108422': 'bull mastiff', 'n02104029': 'kuvasz', 'n02110958': 'pug, pug-dog', 'n02099429': 'curly-coated retriever', 'n02094258': 'Norwich terrier', 'n02112350': 'keeshond', 'n02095570': 'Lakeland terrier', 'n02097209': 'standard schnauzer', 'n02097474': 'Tibetan terrier, chrysanthemum dog', 'n02095314': 'wire-haired fox terrier', 'n02088238': 'basset, basset hound', 'n02112137': 'chow, chow chow', 'n02093428': 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 'n02105855': 'Shetland sheepdog, Shetland sheep dog, Shetland', 'n02111500': 'Great Pyrenees', 'n02085620': 'Chihuahua', 'n02099712': 'Labrador retriever', 'n02111889': 'Samoyed, Samoyede', 'n02088632': 'bluetick', 'n02105412': 'kelpie', 'n02107312': 'miniature pinscher', 'n02091032': 'Italian greyhound', 'n02102318': 'cocker spaniel, English cocker spaniel, cocker', 'n02102480': 'Sussex spaniel', 'n02113023': 'Pembroke, Pembroke Welsh corgi', 'n02086646': 'Blenheim spaniel', 'n02091244': 'Ibizan hound, Ibizan Podenco', 'n02089973': 'English foxhound', 'n02105251': 'briard', 'n02093754': 'Border terrier'}
# train_data['class_name'].unique()[0]
# imagenet_cls_map[train_data['class_name'].unique()[0]]
# mini_imagenet_cls_map={}
# for name in train_data['class_name'].unique():
#     mini_imagenet_cls_map[name] = imagenet_cls_map[name]

mini_imagenet_cls_map = {'n01872401': 'echidna, spiny anteater, anteater',
 'n02417914': 'ibex, Capra ibex',
 'n02106166': 'Border collie',
 'n04235860': 'sleeping bag',
 'n02056570': 'king penguin, Aptenodytes patagonica',
 'n07734744': 'mushroom',
 'n02098286': 'West Highland white terrier',
 'n02097298': 'Scotch terrier, Scottish terrier, Scottie',
 'n02403003': 'ox',
 'n04456115': 'torch',
 'n02408429': 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
 'n09472597': 'volcano',
 'n04004767': 'printer',
 'n03832673': 'notebook, notebook computer',
 'n01748264': 'Indian cobra, Naja naja',
 'n02096437': 'Dandie Dinmont, Dandie Dinmont terrier',
 'n02325366': 'wood rabbit, cottontail, cottontail rabbit',
 'n03857828': 'oscilloscope, scope, cathode-ray oscilloscope, CRO',
 'n03481172': 'hammer',
 'n02701002': 'ambulance',
 'n01855032': 'red-breasted merganser, Mergus serrator',
 'n01698640': 'American alligator, Alligator mississipiensis',
 'n02114548': 'white wolf, Arctic wolf, Canis lupus tundrarum',
 'n01644900': 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
 'n02107574': 'Greater Swiss Mountain dog',
 'n03803284': 'muzzle',
 'n02494079': 'squirrel monkey, Saimiri sciureus',
 'n02027492': 'red-backed sandpiper, dunlin, Erolia alpina',
 'n04296562': 'stage',
 'n03584829': 'iron, smoothing iron',
 'n01843065': 'jacamar',
 'n03530642': 'honeycomb',
 'n02791124': 'barber chair',
 'n04486054': 'triumphal arch',
 'n01744401': 'rock python, rock snake, Python sebae',
 'n03063689': 'coffeepot',
 'n02110958': 'pug, pug-dog',
 'n04507155': 'umbrella',
 'n03710193': 'mailbox, letter box',
 'n01580077': 'jay',
 'n13052670': 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',
 'n02279972': 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
 'n04336792': 'stretcher',
 'n02108915': 'French bulldog',
 'n04517823': 'vacuum, vacuum cleaner',
 'n07753592': 'banana',
 'n02992211': 'cello, violoncello',
 'n01531178': 'goldfinch, Carduelis carduelis',
 'n02396427': 'wild boar, boar, Sus scrofa',
 'n03444034': 'go-kart',
 'n01614925': 'bald eagle, American eagle, Haliaeetus leucocephalus',
 'n04039381': 'racket, racquet',
 'n03888605': 'parallel bars, bars',
 'n03425413': 'gas pump, gasoline pump, petrol pump, island dispenser',
 'n03895866': 'passenger car, coach, carriage',
 'n12998815': 'agaric',
 'n02087394': 'Rhodesian ridgeback',
 'n02097209': 'standard schnauzer',
 'n04259630': 'sombrero',
 'n03445777': 'golf ball',
 'n04040759': 'radiator',
 'n02454379': 'armadillo',
 'n02971356': 'carton',
 'n03929660': 'pick, plectrum, plectron',
 'n02690373': 'airliner',
 'n01774384': 'black widow, Latrodectus mactans',
 'n03134739': 'croquet ball',
 'n02085782': 'Japanese spaniel',
 'n04404412': 'television, television system',
 'n01514668': 'cock',
 'n04525305': 'vending machine',
 'n04560804': 'water jug',
 'n03642806': 'laptop, laptop computer',
 'n02422699': 'impala, Aepyceros melampus',
 'n01985128': 'crayfish, crawfish, crawdad, crawdaddy',
 'n04344873': 'studio couch, day bed',
 'n07716906': 'spaghetti squash',
 'n02951585': 'can opener, tin opener',
 'n03874599': 'padlock',
 'n01753488': 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
 'n02643566': 'lionfish',
 'n04081281': 'restaurant, eating house, eating place, eatery',
 'n02110806': 'basenji',
 'n02009912': 'American egret, great white heron, Egretta albus',
 'n01494475': 'hammerhead, hammerhead shark',
 'n02445715': 'skunk, polecat, wood pussy',
 'n10565667': 'scuba diver',
 'n03355925': 'flagpole, flagstaff',
 'n04204347': 'shopping cart',
 'n04591157': 'Windsor tie',
 'n03781244': 'monastery',
 'n04026417': 'purse',
 'n09288635': 'geyser',
 'n02113624': 'toy poodle',
 'n02113023': 'Pembroke, Pembroke Welsh corgi',
 'n01843383': 'toucan',
 'n04141076': 'sax, saxophone',
 'n03345487': 'fire engine, fire truck',
 'n01983481': 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
 'n01950731': 'sea slug, nudibranch',
 'n02092339': 'Weimaraner',
 'n01729322': 'hognose snake, puff adder, sand viper',
 'n03131574': 'crib, cot',
 'n04606251': 'wreck',
 'n02102177': 'Welsh springer spaniel',
 'n01616318': 'vulture',
 'n04350905': 'suit, suit of clothes',
 'n01532829': 'house finch, linnet, Carpodacus mexicanus',
 'n02321529': 'sea cucumber, holothurian',
 'n01601694': 'water ouzel, dipper',
 'n04127249': 'safety pin',
 'n03598930': 'jigsaw puzzle',
 'n02837789': 'bikini, two-piece',
 'n09193705': 'alp',
 'n02364673': 'guinea pig, Cavia cobaya',
 'n03063599': 'coffee mug',
 'n07248320': 'book jacket, dust cover, dust jacket, dust wrapper',
 'n03109150': 'corkscrew, bottle screw',
 'n01688243': 'frilled lizard, Chlamydosaurus kingi',
 'n02002556': 'white stork, Ciconia ciconia',
 'n03770439': 'miniskirt, mini',
 'n04310018': 'steam locomotive',
 'n03445924': 'golfcart, golf cart',
 'n02101388': 'Brittany spaniel',
 'n04037443': 'racer, race car, racing car',
 'n04409515': 'tennis ball',
 'n03188531': 'diaper, nappy, napkin',
 'n04120489': 'running shoe',
 'n01806567': 'quail',
 'n02109047': 'Great Dane',
 'n04131690': 'saltshaker, salt shaker',
 'n01873310': 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
 'n04540053': 'volleyball',
 'n02085620': 'Chihuahua',
 'n04275548': "spider web, spider's web",
 'n02641379': 'gar, garfish, garpike, billfish, Lepisosteus osseus',
 'n02268443': "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
 'n01534433': 'junco, snowbird',
 'n02105056': 'groenendael',
 'n02769748': 'backpack, back pack, knapsack, packsack, rucksack, haversack',
 'n01774750': 'tarantula',
 'n02410509': 'bison',
 'n03763968': 'military uniform',
 'n03691459': 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system',
 'n01882714': 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
 'n09256479': 'coral reef',
 'n02109525': 'Saint Bernard, St Bernard',
 'n02105412': 'kelpie',
 'n02437312': 'Arabian camel, dromedary, Camelus dromedarius',
 'n02939185': 'caldron, cauldron',
 'n03788365': 'mosquito net',
 'n03673027': 'liner, ocean liner',
 'n02105855': 'Shetland sheepdog, Shetland sheep dog, Shetland',
 'n03291819': 'envelope',
 'n01749939': 'green mamba',
 'n04325704': 'stole',
 'n02415577': 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
 'n03376595': 'folding chair',
 'n02793495': 'barn',
 'n02018795': 'bustard',
 'n04553703': 'washbasin, handbasin, washbowl, lavabo, wash-hand basin',
 'n04317175': 'stethoscope',
 'n04380533': 'table lamp',
 'n01833805': 'hummingbird',
 'n02895154': 'breastplate, aegis, egis',
 'n02389026': 'sorrel',
 'n03942813': 'ping-pong ball',
 'n02094114': 'Norfolk terrier',
 'n02129604': 'tiger, Panthera tigris',
 'n04428191': 'thresher, thrasher, threshing machine',
 'n03916031': 'perfume, essence',
 'n04326547': 'stone wall',
 'n02892201': 'brass, memorial tablet, plaque',
 'n04005630': 'prison, prison house',
 'n07717410': 'acorn squash',
 'n04252225': 'snowplow, snowplough',
 'n04554684': 'washer, automatic washer, washing machine',
 'n02443484': 'black-footed ferret, ferret, Mustela nigripes',
 'n01818515': 'macaw',
 'n02672831': 'accordion, piano accordion, squeeze box',
 'n03773504': 'missile',
 'n02113799': 'standard poodle',
 'n02100735': 'English setter',
 'n02104029': 'kuvasz',
 'n04370456': 'sweatshirt',
 'n07579787': 'plate',
 'n04447861': 'toilet seat',
 'n03759954': 'microphone, mike',
 'n02112350': 'keeshond',
 'n04162706': 'seat belt, seatbelt',
 'n02100236': 'German short-haired pointer',
 'n01751748': 'sea snake',
 'n02133161': 'American black bear, black bear, Ursus americanus, Euarctos americanus',
 'n02236044': 'mantis, mantid',
 'n01592084': 'chickadee',
 'n04604644': 'worm fence, snake fence, snake-rail fence, Virginia fence',
 'n02011460': 'bittern',
 'n02097047': 'miniature schnauzer',
 'n12768682': 'buckeye, horse chestnut, conker',
 'n03871628': 'packet',
 'n07716358': 'zucchini, courgette',
 'n03933933': 'pier',
 'n04479046': 'trench coat',
 'n03089624': 'confectionery, confectionary, candy store',
 'n03980874': 'poncho',
 'n03127925': 'crate',
 'n04209239': 'shower curtain',
 'n04136333': 'sarong',
 'n02443114': 'polecat, fitch, foulmart, foumart, Mustela putorius',
 'n04505470': 'typewriter keyboard',
 'n02094258': 'Norwich terrier',
 'n04229816': 'ski mask',
 'n02089867': 'Walker hound, Walker foxhound',
 'n02092002': 'Scottish deerhound, deerhound',
 'n07920052': 'espresso',
 'n03347037': 'fire screen, fireguard',
 'n02099267': 'flat-coated retriever',
 'n02802426': 'basketball',
 'n01739381': 'vine snake',
 'n03544143': 'hourglass',
 'n02096585': 'Boston bull, Boston terrier',
 'n02105162': 'malinois',
 'n04493381': 'tub, vat',
 'n02708093': 'analog clock',
 'n03891251': 'park bench',
 'n04208210': 'shovel',
 'n04033901': 'quill, quill pen',
 'n02088632': 'bluetick',
 'n04550184': 'wardrobe, closet, press',
 'n04596742': 'wok',
 'n03976657': 'pole',
 'n02091134': 'whippet',
 'n07754684': 'jackfruit, jak, jack',
 'n01978287': 'Dungeness crab, Cancer magister',
 'n02727426': 'apiary, bee house',
 'n02115913': 'dhole, Cuon alpinus',
 'n04263257': 'soup bowl',
 'n03476684': 'hair slide',
 'n04476259': 'tray',
 'n02077923': 'sea lion',
 'n03794056': 'mousetrap',
 'n01728572': 'thunder snake, worm snake, Carphophis amoenus',
 'n04579432': 'whistle',
 'n02992529': 'cellular telephone, cellular phone, cellphone, cell, mobile phone',
 'n13054560': 'bolete',
 'n07860988': 'dough',
 'n03207941': 'dishwasher, dish washer, dishwashing machine',
 'n07615774': 'ice lolly, lolly, lollipop, popsicle',
 'n02102480': 'Sussex spaniel',
 'n01910747': 'jellyfish',
 'n04532106': 'vestment',
 'n04266014': 'space shuttle',
 'n04125021': 'safe',
 'n01530575': 'brambling, Fringilla montifringilla',
 'n04599235': 'wool, woolen, woollen',
 'n04435653': 'tile roof',
 'n02097130': 'giant schnauzer',
 'n02114712': 'red wolf, maned wolf, Canis rufus, Canis niger',
 'n03271574': 'electric fan, blower',
 'n03724870': 'mask',
 'n02096177': 'cairn, cairn terrier',
 'n04509417': 'unicycle, monocycle',
 'n01828970': 'bee eater',
 'n02100583': 'vizsla, Hungarian pointer',
 'n03496892': 'harvester, reaper',
 'n04209133': 'shower cap',
 'n02917067': 'bullet train, bullet',
 'n02906734': 'broom',
 'n02749479': 'assault rifle, assault gun',
 'n01860187': 'black swan, Cygnus atratus',
 'n01537544': 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
 'n03637318': 'lampshade, lamp shade',
 'n02132136': 'brown bear, bruin, Ursus arctos',
 'n02088094': 'Afghan hound, Afghan',
 'n03379051': 'football helmet',
 'n04201297': 'shoji',
 'n01855672': 'goose',
 'n01632777': 'axolotl, mud puppy, Ambystoma mexicanum',
 'n03249569': 'drum, membranophone, tympan',
 'n04487394': 'trombone',
 'n02892767': 'brassiere, bra, bandeau',
 'n04146614': 'school bus',
 'n02441942': 'weasel',
 'n07873807': 'pizza, pizza pie',
 'n02091467': 'Norwegian elkhound, elkhound',
 'n01807496': 'partridge',
 'n02808440': 'bathtub, bathing tub, bath, tub',
 'n02088238': 'basset, basset hound',
 'n02110185': 'Siberian husky',
 'n01641577': 'bullfrog, Rana catesbeiana',
 'n01770393': 'scorpion',
 'n02090379': 'redbone',
 'n02457408': 'three-toed sloth, ai, Bradypus tridactylus',
 'n04141975': 'scale, weighing machine',
 'n01795545': 'black grouse',
 'n03958227': 'plastic bag',
 'n04485082': 'tripod',
 'n04019541': 'puck, hockey puck',
 'n02094433': 'Yorkshire terrier',
 'n04355338': 'sundial',
 'n01695060': 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
 'n03956157': 'planetarium',
 'n01697457': 'African crocodile, Nile crocodile, Crocodylus niloticus',
 'n02172182': 'dung beetle',
 'n03388043': 'fountain',
 'n01518878': 'ostrich, Struthio camelus',
 'n03995372': 'power drill',
 'n04589890': 'window screen',
 'n04254777': 'sock',
 'n04584207': 'wig',
 'n04591713': 'wine bottle',
 'n04118776': 'rule, ruler',
 'n02091032': 'Italian greyhound',
 'n04429376': 'throne',
 'n02493793': 'spider monkey, Ateles geoffroyi',
 'n02999410': 'chain',
 'n10148035': 'groom, bridegroom',
 'n03124043': 'cowboy boot',
 'n02687172': 'aircraft carrier, carrier, flattop, attack aircraft carrier',
 'n02493509': 'titi, titi monkey',
 'n01775062': 'wolf spider, hunting spider',
 'n04192698': 'shield, buckler',
 'n02058221': 'albatross, mollymawk',
 'n03595614': 'jersey, T-shirt, tee shirt',
 'n04523525': 'vault',
 'n03814906': 'necklace',
 'n02412080': 'ram, tup',
 'n09835506': 'ballplayer, baseball player',
 'n04074963': 'remote control, remote',
 'n02099601': 'golden retriever',
 'n02100877': 'Irish setter, red setter',
 'n01740131': 'night snake, Hypsiglena torquata',
 'n01608432': 'kite',
 'n02037110': 'oystercatcher, oyster catcher',
 'n02488702': 'colobus, colobus monkey',
 'n02108089': 'boxer',
 'n04376876': 'syringe',
 'n02814533': 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
 'n07742313': 'Granny Smith',
 'n02111277': 'Newfoundland, Newfoundland dog',
 'n03692522': "loupe, jeweler's loupe",
 'n07718747': 'artichoke, globe artichoke',
 'n04522168': 'vase',
 'n04049303': 'rain barrel',
 'n02106550': 'Rottweiler',
 'n04418357': 'theater curtain, theatre curtain',
 'n02088364': 'beagle',
 'n03676483': 'lipstick, lip rouge',
 'n04372370': 'switch, electric switch, electrical switch',
 'n02795169': 'barrel, cask',
 'n02510455': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
 'n02974003': 'car wheel',
 'n02281787': 'lycaenid, lycaenid butterfly',
 'n01677366': 'common iguana, iguana, Iguana iguana',
 'n01773797': 'garden spider, Aranea diademata',
 'n02123045': 'tabby, tabby cat',
 'n01968897': 'chambered nautilus, pearly nautilus, nautilus',
 'n04465501': 'tractor',
 'n03903868': 'pedestal, plinth, footstall',
 'n02799071': 'baseball',
 'n06785654': 'crossword puzzle, crossword',
 'n02699494': 'altar',
 'n02129165': 'lion, king of beasts, Panthera leo',
 'n07583066': 'guacamole',
 'n03775071': 'mitten',
 'n03761084': 'microwave, microwave oven',
 'n02814860': 'beacon, lighthouse, beacon light, pharos',
 'n03649909': 'lawn mower, mower',
 'n02093256': 'Staffordshire bullterrier, Staffordshire bull terrier',
 'n02130308': 'cheetah, chetah, Acinonyx jubatus',
 'n02102040': 'English springer, English springer spaniel',
 'n04536866': 'violin, fiddle',
 'n02730930': 'apron',
 'n11939491': 'daisy',
 'n02487347': 'macaque',
 'n02102318': 'cocker spaniel, English cocker spaniel, cocker',
 'n07714990': 'broccoli',
 'n04612504': 'yawl',
 'n03998194': 'prayer rug, prayer mat',
 'n01770081': 'harvestman, daddy longlegs, Phalangium opilio',
 'n02134084': 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
 'n02927161': 'butcher shop, meat market',
 'n04069434': 'reflex camera',
 'n03787032': 'mortarboard',
 'n02111889': 'Samoyed, Samoyede',
 'n03793489': 'mouse, computer mouse',
 'n07875152': 'potpie',
 'n02009229': 'little blue heron, Egretta caerulea',
 'n02086240': 'Shih-Tzu',
 'n07715103': 'cauliflower',
 'n02676566': 'acoustic guitar',
 'n04443257': 'tobacco shop, tobacconist shop, tobacconist',
 'n04483307': 'trimaran',
 'n03630383': 'lab coat, laboratory coat',
 'n02326432': 'hare',
 'n02124075': 'Egyptian cat',
 'n02280649': 'cabbage butterfly',
 'n02361337': 'marmot',
 'n02692877': 'airship, dirigible',
 'n04557648': 'water bottle',
 'n12267677': 'acorn',
 'n02165456': 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
 'n02112137': 'chow, chow chow',
 'n01914609': 'sea anemone, anemone',
 'n01735189': 'garter snake, grass snake',
 'n01944390': 'snail',
 'n03498962': 'hatchet',
 'n04356056': 'sunglasses, dark glasses, shades',
 'n02089973': 'English foxhound',
 'n02123597': 'Siamese cat, Siamese',
 'n04254680': 'soccer ball',
 'n02111500': 'Great Pyrenees',
 'n03394916': 'French horn, horn',
 'n07745940': 'strawberry',
 'n03709823': 'mailbag, postbag',
 'n07614500': 'ice cream, icecream',
 'n01704323': 'triceratops',
 'n03792782': 'mountain bike, all-terrain bike, off-roader',
 'n02883205': 'bow tie, bow-tie, bowtie',
 'n02101006': 'Gordon setter',
 'n01644373': 'tree frog, tree-frog',
 'n04366367': 'suspension bridge',
 'n02879718': 'bow',
 'n12144580': 'corn',
 'n07613480': 'trifle',
 'n04371430': 'swimming trunks, bathing trunks',
 'n03534580': 'hoopskirt, crinoline',
 'n03450230': 'gown',
 'n03938244': 'pillow',
 'n01443537': 'goldfish, Carassius auratus',
 'n02086079': 'Pekinese, Pekingese, Peke',
 'n02391049': 'zebra',
 'n02398521': 'hippopotamus, hippo, river horse, Hippopotamus amphibius',
 'n02093428': 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
 'n02107142': 'Doberman, Doberman pinscher',
 'n03208938': 'disk brake, disc brake',
 'n04399382': 'teddy, teddy bear',
 'n04118538': 'rugby ball',
 'n04355933': 'sunglass',
 'n04330267': 'stove',
 'n03045698': 'cloak',
 'n02128385': 'leopard, Panthera pardus',
 'n04442312': 'toaster',
 'n02356798': 'fox squirrel, eastern fox squirrel, Sciurus niger',
 'n03970156': "plunger, plumber's helper",
 'n02107683': 'Bernese mountain dog',
 'n03982430': 'pool table, billiard table, snooker table',
 'n02342885': 'hamster',
 'n09246464': 'cliff, drop, drop-off',
 'n02114367': 'timber wolf, grey wolf, gray wolf, Canis lupus',
 'n07747607': 'orange',
 'n02113712': 'miniature poodle',
 'n04487081': 'trolleybus, trolley coach, trackless trolley',
 'n04238763': 'slide rule, slipstick',
 'n02088466': 'bloodhound, sleuthhound',
 'n01630670': 'common newt, Triturus vulgaris',
 'n02117135': 'hyena, hyaena',
 'n04501370': 'turnstile',
 'n02098413': 'Lhasa, Lhasa apso',
 'n02127052': 'lynx, catamount',
 'n02782093': 'balloon',
 'n04548362': 'wallet, billfold, notecase, pocketbook',
 'n01496331': 'electric ray, crampfish, numbfish, torpedo',
 'n02841315': 'binoculars, field glasses, opera glasses',
 'n02480495': 'orangutan, orang, orangutang, Pongo pygmaeus',
 'n03961711': 'plate rack',
 'n02106030': 'collie',
 'n02397096': 'warthog',
 'n04041544': 'radio, wireless',
 'n02865351': 'bolo tie, bolo, bola tie, bola',
 'n03297495': 'espresso maker',
 'n01742172': 'boa constrictor, Constrictor constrictor',
 'n04252077': 'snowmobile',
 'n02087046': 'toy terrier',
 'n07768694': 'pomegranate',
 'n01582220': 'magpie',
 'n01930112': 'nematode, nematode worm, roundworm',
 'n01558993': 'robin, American robin, Turdus migratorius',
 'n02442845': 'mink',
 'n02105641': 'Old English sheepdog, bobtail',
 'n02120079': 'Arctic fox, white fox, Alopex lagopus',
 'n02085936': 'Maltese dog, Maltese terrier, Maltese',
 'n01694178': 'African chameleon, Chamaeleo chamaeleon',
 'n04371774': 'swing',
 'n03447721': 'gong, tam-tam',
 'n04251144': 'snorkel',
 'n01484850': 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
 'n02007558': 'flamingo',
 'n07753113': 'fig',
 'n02277742': 'ringlet, ringlet butterfly',
 'n03720891': 'maraca',
 'n02051845': 'pelican',
 'n02090622': 'borzoi, Russian wolfhound',
 'n02091831': 'Saluki, gazelle hound',
 'n02843684': 'birdhouse',
 'n04357314': 'sunscreen, sunblock, sun blocker',
 'n02437616': 'llama',
 'n02108422': 'bull mastiff',
 'n03729826': 'matchstick',
 'n02870880': 'bookcase'}

'''
{'n01872401': 'a photo of echidna, spiny anteater, anteater',
 'n02417914': 'a photo of ibex, Capra ibex',
 'n02106166': 'a photo of Border collie',
 'n04235860': 'a photo of sleeping bag',
 'n02056570': 'a photo of king penguin, Aptenodytes patagonica',
 'n07734744': 'a photo of mushroom',
 'n02098286': 'a photo of West Highland white terrier',
 'n02097298': 'a photo of Scotch terrier, Scottish terrier, Scottie',
 'n02403003': 'a photo of ox',
 'n04456115': 'a photo of torch',
 'n02408429': 'a photo of water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
 'n09472597': 'a photo of volcano',
 'n04004767': 'a photo of printer',
 'n03832673': 'a photo of notebook, notebook computer',
 'n01748264': 'a photo of Indian cobra, Naja naja',
 'n02096437': 'a photo of Dandie Dinmont, Dandie Dinmont terrier',
 'n02325366': 'a photo of wood rabbit, cottontail, cottontail rabbit',
 'n03857828': 'a photo of oscilloscope, scope, cathode-ray oscilloscope, CRO',
 'n03481172': 'a photo of hammer',
 'n02701002': 'a photo of ambulance',
 'n01855032': 'a photo of red-breasted merganser, Mergus serrator',
 'n01698640': 'a photo of American alligator, Alligator mississipiensis',
 'n02114548': 'a photo of white wolf, Arctic wolf, Canis lupus tundrarum',
 'n01644900': 'a photo of tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
 'n02107574': 'a photo of Greater Swiss Mountain dog',
 'n03803284': 'a photo of muzzle',
 'n02494079': 'a photo of squirrel monkey, Saimiri sciureus',
 'n02027492': 'a photo of red-backed sandpiper, dunlin, Erolia alpina',
 'n04296562': 'a photo of stage',
 'n03584829': 'a photo of iron, smoothing iron',
 'n01843065': 'a photo of jacamar',
 'n03530642': 'a photo of honeycomb',
 'n02791124': 'a photo of barber chair',
 'n04486054': 'a photo of triumphal arch',
 'n01744401': 'a photo of rock python, rock snake, Python sebae',
 'n03063689': 'a photo of coffeepot',
 'n02110958': 'a photo of pug, pug-dog',
 'n04507155': 'a photo of umbrella',
 'n03710193': 'a photo of mailbox, letter box',
 'n01580077': 'a photo of jay',
 'n13052670': 'a photo of hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',
 'n02279972': 'a photo of monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
 'n04336792': 'a photo of stretcher',
 'n02108915': 'a photo of French bulldog',
 'n04517823': 'a photo of vacuum, vacuum cleaner',
 'n07753592': 'a photo of banana',
 'n02992211': 'a photo of cello, violoncello',
 'n01531178': 'a photo of goldfinch, Carduelis carduelis',
 'n02396427': 'a photo of wild boar, boar, Sus scrofa',
 'n03444034': 'a photo of go-kart',
 'n01614925': 'a photo of bald eagle, American eagle, Haliaeetus leucocephalus',
 'n04039381': 'a photo of racket, racquet',
 'n03888605': 'a photo of parallel bars, bars',
 'n03425413': 'a photo of gas pump, gasoline pump, petrol pump, island dispenser',
 'n03895866': 'a photo of passenger car, coach, carriage',
 'n12998815': 'a photo of agaric',
 'n02087394': 'a photo of Rhodesian ridgeback',
 'n02097209': 'a photo of standard schnauzer',
 'n04259630': 'a photo of sombrero',
 'n03445777': 'a photo of golf ball',
 'n04040759': 'a photo of radiator',
 'n02454379': 'a photo of armadillo',
 'n02971356': 'a photo of carton',
 'n03929660': 'a photo of pick, plectrum, plectron',
 'n02690373': 'a photo of airliner',
 'n01774384': 'a photo of black widow, Latrodectus mactans',
 'n03134739': 'a photo of croquet ball',
 'n02085782': 'a photo of Japanese spaniel',
 'n04404412': 'a photo of television, television system',
 'n01514668': 'a photo of cock',
 'n04525305': 'a photo of vending machine',
 'n04560804': 'a photo of water jug',
 'n03642806': 'a photo of laptop, laptop computer',
 'n02422699': 'a photo of impala, Aepyceros melampus',
 'n01985128': 'a photo of crayfish, crawfish, crawdad, crawdaddy',
 'n04344873': 'a photo of studio couch, day bed',
 'n07716906': 'a photo of spaghetti squash',
 'n02951585': 'a photo of can opener, tin opener',
 'n03874599': 'a photo of padlock',
 'n01753488': 'a photo of horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
 'n02643566': 'a photo of lionfish',
 'n04081281': 'a photo of restaurant, eating house, eating place, eatery',
 'n02110806': 'a photo of basenji',
 'n02009912': 'a photo of American egret, great white heron, Egretta albus',
 'n01494475': 'a photo of hammerhead, hammerhead shark',
 'n02445715': 'a photo of skunk, polecat, wood pussy',
 'n10565667': 'a photo of scuba diver',
 'n03355925': 'a photo of flagpole, flagstaff',
 'n04204347': 'a photo of shopping cart',
 'n04591157': 'a photo of Windsor tie',
 'n03781244': 'a photo of monastery',
 'n04026417': 'a photo of purse',
 'n09288635': 'a photo of geyser',
 'n02113624': 'a photo of toy poodle',
 'n02113023': 'a photo of Pembroke, Pembroke Welsh corgi',
 'n01843383': 'a photo of toucan',
 'n04141076': 'a photo of sax, saxophone',
 'n03345487': 'a photo of fire engine, fire truck',
 'n01983481': 'a photo of American lobster, Northern lobster, Maine lobster, Homarus americanus',
 'n01950731': 'a photo of sea slug, nudibranch',
 'n02092339': 'a photo of Weimaraner',
 'n01729322': 'a photo of hognose snake, puff adder, sand viper',
 'n03131574': 'a photo of crib, cot',
 'n04606251': 'a photo of wreck',
 'n02102177': 'a photo of Welsh springer spaniel',
 'n01616318': 'a photo of vulture',
 'n04350905': 'a photo of suit, suit of clothes',
 'n01532829': 'a photo of house finch, linnet, Carpodacus mexicanus',
 'n02321529': 'a photo of sea cucumber, holothurian',
 'n01601694': 'a photo of water ouzel, dipper',
 'n04127249': 'a photo of safety pin',
 'n03598930': 'a photo of jigsaw puzzle',
 'n02837789': 'a photo of bikini, two-piece',
 'n09193705': 'a photo of alp',
 'n02364673': 'a photo of guinea pig, Cavia cobaya',
 'n03063599': 'a photo of coffee mug',
 'n07248320': 'a photo of book jacket, dust cover, dust jacket, dust wrapper',
 'n03109150': 'a photo of corkscrew, bottle screw',
 'n01688243': 'a photo of frilled lizard, Chlamydosaurus kingi',
 'n02002556': 'a photo of white stork, Ciconia ciconia',
 'n03770439': 'a photo of miniskirt, mini',
 'n04310018': 'a photo of steam locomotive',
 'n03445924': 'a photo of golfcart, golf cart',
 'n02101388': 'a photo of Brittany spaniel',
 'n04037443': 'a photo of racer, race car, racing car',
 'n04409515': 'a photo of tennis ball',
 'n03188531': 'a photo of diaper, nappy, napkin',
 'n04120489': 'a photo of running shoe',
 'n01806567': 'a photo of quail',
 'n02109047': 'a photo of Great Dane',
 'n04131690': 'a photo of saltshaker, salt shaker',
 'n01873310': 'a photo of platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
 'n04540053': 'a photo of volleyball',
 'n02085620': 'a photo of Chihuahua',
 'n04275548': "a photo of spider web, spider's web",
 'n02641379': 'a photo of gar, garfish, garpike, billfish, Lepisosteus osseus',
 'n02268443': "a photo of dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
 'n01534433': 'a photo of junco, snowbird',
 'n02105056': 'a photo of groenendael',
 'n02769748': 'a photo of backpack, back pack, knapsack, packsack, rucksack, haversack',
 'n01774750': 'a photo of tarantula',
 'n02410509': 'a photo of bison',
 'n03763968': 'a photo of military uniform',
 'n03691459': 'a photo of loudspeaker, speaker, speaker unit, loudspeaker system, speaker system',
 'n01882714': 'a photo of koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
 'n09256479': 'a photo of coral reef',
 'n02109525': 'a photo of Saint Bernard, St Bernard',
 'n02105412': 'a photo of kelpie',
 'n02437312': 'a photo of Arabian camel, dromedary, Camelus dromedarius',
 'n02939185': 'a photo of caldron, cauldron',
 'n03788365': 'a photo of mosquito net',
 'n03673027': 'a photo of liner, ocean liner',
 'n02105855': 'a photo of Shetland sheepdog, Shetland sheep dog, Shetland',
 'n03291819': 'a photo of envelope',
 'n01749939': 'a photo of green mamba',
 'n04325704': 'a photo of stole',
 'n02415577': 'a photo of bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
 'n03376595': 'a photo of folding chair',
 'n02793495': 'a photo of barn',
 'n02018795': 'a photo of bustard',
 'n04553703': 'a photo of washbasin, handbasin, washbowl, lavabo, wash-hand basin',
 'n04317175': 'a photo of stethoscope',
 'n04380533': 'a photo of table lamp',
 'n01833805': 'a photo of hummingbird',
 'n02895154': 'a photo of breastplate, aegis, egis',
 'n02389026': 'a photo of sorrel',
 'n03942813': 'a photo of ping-pong ball',
 'n02094114': 'a photo of Norfolk terrier',
 'n02129604': 'a photo of tiger, Panthera tigris',
 'n04428191': 'a photo of thresher, thrasher, threshing machine',
 'n03916031': 'a photo of perfume, essence',
 'n04326547': 'a photo of stone wall',
 'n02892201': 'a photo of brass, memorial tablet, plaque',
 'n04005630': 'a photo of prison, prison house',
 'n07717410': 'a photo of acorn squash',
 'n04252225': 'a photo of snowplow, snowplough',
 'n04554684': 'a photo of washer, automatic washer, washing machine',
 'n02443484': 'a photo of black-footed ferret, ferret, Mustela nigripes',
 'n01818515': 'a photo of macaw',
 'n02672831': 'a photo of accordion, piano accordion, squeeze box',
 'n03773504': 'a photo of missile',
 'n02113799': 'a photo of standard poodle',
 'n02100735': 'a photo of English setter',
 'n02104029': 'a photo of kuvasz',
 'n04370456': 'a photo of sweatshirt',
 'n07579787': 'a photo of plate',
 'n04447861': 'a photo of toilet seat',
 'n03759954': 'a photo of microphone, mike',
 'n02112350': 'a photo of keeshond',
 'n04162706': 'a photo of seat belt, seatbelt',
 'n02100236': 'a photo of German short-haired pointer',
 'n01751748': 'a photo of sea snake',
 'n02133161': 'a photo of American black bear, black bear, Ursus americanus, Euarctos americanus',
 'n02236044': 'a photo of mantis, mantid',
 'n01592084': 'a photo of chickadee',
 'n04604644': 'a photo of worm fence, snake fence, snake-rail fence, Virginia fence',
 'n02011460': 'a photo of bittern',
 'n02097047': 'a photo of miniature schnauzer',
 'n12768682': 'a photo of buckeye, horse chestnut, conker',
 'n03871628': 'a photo of packet',
 'n07716358': 'a photo of zucchini, courgette',
 'n03933933': 'a photo of pier',
 'n04479046': 'a photo of trench coat',
 'n03089624': 'a photo of confectionery, confectionary, candy store',
 'n03980874': 'a photo of poncho',
 'n03127925': 'a photo of crate',
 'n04209239': 'a photo of shower curtain',
 'n04136333': 'a photo of sarong',
 'n02443114': 'a photo of polecat, fitch, foulmart, foumart, Mustela putorius',
 'n04505470': 'a photo of typewriter keyboard',
 'n02094258': 'a photo of Norwich terrier',
 'n04229816': 'a photo of ski mask',
 'n02089867': 'a photo of Walker hound, Walker foxhound',
 'n02092002': 'a photo of Scottish deerhound, deerhound',
 'n07920052': 'a photo of espresso',
 'n03347037': 'a photo of fire screen, fireguard',
 'n02099267': 'a photo of flat-coated retriever',
 'n02802426': 'a photo of basketball',
 'n01739381': 'a photo of vine snake',
 'n03544143': 'a photo of hourglass',
 'n02096585': 'a photo of Boston bull, Boston terrier',
 'n02105162': 'a photo of malinois',
 'n04493381': 'a photo of tub, vat',
 'n02708093': 'a photo of analog clock',
 'n03891251': 'a photo of park bench',
 'n04208210': 'a photo of shovel',
 'n04033901': 'a photo of quill, quill pen',
 'n02088632': 'a photo of bluetick',
 'n04550184': 'a photo of wardrobe, closet, press',
 'n04596742': 'a photo of wok',
 'n03976657': 'a photo of pole',
 'n02091134': 'a photo of whippet',
 'n07754684': 'a photo of jackfruit, jak, jack',
 'n01978287': 'a photo of Dungeness crab, Cancer magister',
 'n02727426': 'a photo of apiary, bee house',
 'n02115913': 'a photo of dhole, Cuon alpinus',
 'n04263257': 'a photo of soup bowl',
 'n03476684': 'a photo of hair slide',
 'n04476259': 'a photo of tray',
 'n02077923': 'a photo of sea lion',
 'n03794056': 'a photo of mousetrap',
 'n01728572': 'a photo of thunder snake, worm snake, Carphophis amoenus',
 'n04579432': 'a photo of whistle',
 'n02992529': 'a photo of cellular telephone, cellular phone, cellphone, cell, mobile phone',
 'n13054560': 'a photo of bolete',
 'n07860988': 'a photo of dough',
 'n03207941': 'a photo of dishwasher, dish washer, dishwashing machine',
 'n07615774': 'a photo of ice lolly, lolly, lollipop, popsicle',
 'n02102480': 'a photo of Sussex spaniel',
 'n01910747': 'a photo of jellyfish',
 'n04532106': 'a photo of vestment',
 'n04266014': 'a photo of space shuttle',
 'n04125021': 'a photo of safe',
 'n01530575': 'a photo of brambling, Fringilla montifringilla',
 'n04599235': 'a photo of wool, woolen, woollen',
 'n04435653': 'a photo of tile roof',
 'n02097130': 'a photo of giant schnauzer',
 'n02114712': 'a photo of red wolf, maned wolf, Canis rufus, Canis niger',
 'n03271574': 'a photo of electric fan, blower',
 'n03724870': 'a photo of mask',
 'n02096177': 'a photo of cairn, cairn terrier',
 'n04509417': 'a photo of unicycle, monocycle',
 'n01828970': 'a photo of bee eater',
 'n02100583': 'a photo of vizsla, Hungarian pointer',
 'n03496892': 'a photo of harvester, reaper',
 'n04209133': 'a photo of shower cap',
 'n02917067': 'a photo of bullet train, bullet',
 'n02906734': 'a photo of broom',
 'n02749479': 'a photo of assault rifle, assault gun',
 'n01860187': 'a photo of black swan, Cygnus atratus',
 'n01537544': 'a photo of indigo bunting, indigo finch, indigo bird, Passerina cyanea',
 'n03637318': 'a photo of lampshade, lamp shade',
 'n02132136': 'a photo of brown bear, bruin, Ursus arctos',
 'n02088094': 'a photo of Afghan hound, Afghan',
 'n03379051': 'a photo of football helmet',
 'n04201297': 'a photo of shoji',
 'n01855672': 'a photo of goose',
 'n01632777': 'a photo of axolotl, mud puppy, Ambystoma mexicanum',
 'n03249569': 'a photo of drum, membranophone, tympan',
 'n04487394': 'a photo of trombone',
 'n02892767': 'a photo of brassiere, bra, bandeau',
 'n04146614': 'a photo of school bus',
 'n02441942': 'a photo of weasel',
 'n07873807': 'a photo of pizza, pizza pie',
 'n02091467': 'a photo of Norwegian elkhound, elkhound',
 'n01807496': 'a photo of partridge',
 'n02808440': 'a photo of bathtub, bathing tub, bath, tub',
 'n02088238': 'a photo of basset, basset hound',
 'n02110185': 'a photo of Siberian husky',
 'n01641577': 'a photo of bullfrog, Rana catesbeiana',
 'n01770393': 'a photo of scorpion',
 'n02090379': 'a photo of redbone',
 'n02457408': 'a photo of three-toed sloth, ai, Bradypus tridactylus',
 'n04141975': 'a photo of scale, weighing machine',
 'n01795545': 'a photo of black grouse',
 'n03958227': 'a photo of plastic bag',
 'n04485082': 'a photo of tripod',
 'n04019541': 'a photo of puck, hockey puck',
 'n02094433': 'a photo of Yorkshire terrier',
 'n04355338': 'a photo of sundial',
 'n01695060': 'a photo of Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
 'n03956157': 'a photo of planetarium',
 'n01697457': 'a photo of African crocodile, Nile crocodile, Crocodylus niloticus',
 'n02172182': 'a photo of dung beetle',
 'n03388043': 'a photo of fountain',
 'n01518878': 'a photo of ostrich, Struthio camelus',
 'n03995372': 'a photo of power drill',
 'n04589890': 'a photo of window screen',
 'n04254777': 'a photo of sock',
 'n04584207': 'a photo of wig',
 'n04591713': 'a photo of wine bottle',
 'n04118776': 'a photo of rule, ruler',
 'n02091032': 'a photo of Italian greyhound',
 'n04429376': 'a photo of throne',
 'n02493793': 'a photo of spider monkey, Ateles geoffroyi',
 'n02999410': 'a photo of chain',
 'n10148035': 'a photo of groom, bridegroom',
 'n03124043': 'a photo of cowboy boot',
 'n02687172': 'a photo of aircraft carrier, carrier, flattop, attack aircraft carrier',
 'n02493509': 'a photo of titi, titi monkey',
 'n01775062': 'a photo of wolf spider, hunting spider',
 'n04192698': 'a photo of shield, buckler',
 'n02058221': 'a photo of albatross, mollymawk',
 'n03595614': 'a photo of jersey, T-shirt, tee shirt',
 'n04523525': 'a photo of vault',
 'n03814906': 'a photo of necklace',
 'n02412080': 'a photo of ram, tup',
 'n09835506': 'a photo of ballplayer, baseball player',
 'n04074963': 'a photo of remote control, remote',
 'n02099601': 'a photo of golden retriever',
 'n02100877': 'a photo of Irish setter, red setter',
 'n01740131': 'a photo of night snake, Hypsiglena torquata',
 'n01608432': 'a photo of kite',
 'n02037110': 'a photo of oystercatcher, oyster catcher',
 'n02488702': 'a photo of colobus, colobus monkey',
 'n02108089': 'a photo of boxer',
 'n04376876': 'a photo of syringe',
 'n02814533': 'a photo of beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
 'n07742313': 'a photo of Granny Smith',
 'n02111277': 'a photo of Newfoundland, Newfoundland dog',
 'n03692522': "a photo of loupe, jeweler's loupe",
 'n07718747': 'a photo of artichoke, globe artichoke',
 'n04522168': 'a photo of vase',
 'n04049303': 'a photo of rain barrel',
 'n02106550': 'a photo of Rottweiler',
 'n04418357': 'a photo of theater curtain, theatre curtain',
 'n02088364': 'a photo of beagle',
 'n03676483': 'a photo of lipstick, lip rouge',
 'n04372370': 'a photo of switch, electric switch, electrical switch',
 'n02795169': 'a photo of barrel, cask',
 'n02510455': 'a photo of giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
 'n02974003': 'a photo of car wheel',
 'n02281787': 'a photo of lycaenid, lycaenid butterfly',
 'n01677366': 'a photo of common iguana, iguana, Iguana iguana',
 'n01773797': 'a photo of garden spider, Aranea diademata',
 'n02123045': 'a photo of tabby, tabby cat',
 'n01968897': 'a photo of chambered nautilus, pearly nautilus, nautilus',
 'n04465501': 'a photo of tractor',
 'n03903868': 'a photo of pedestal, plinth, footstall',
 'n02799071': 'a photo of baseball',
 'n06785654': 'a photo of crossword puzzle, crossword',
 'n02699494': 'a photo of altar',
 'n02129165': 'a photo of lion, king of beasts, Panthera leo',
 'n07583066': 'a photo of guacamole',
 'n03775071': 'a photo of mitten',
 'n03761084': 'a photo of microwave, microwave oven',
 'n02814860': 'a photo of beacon, lighthouse, beacon light, pharos',
 'n03649909': 'a photo of lawn mower, mower',
 'n02093256': 'a photo of Staffordshire bullterrier, Staffordshire bull terrier',
 'n02130308': 'a photo of cheetah, chetah, Acinonyx jubatus',
 'n02102040': 'a photo of English springer, English springer spaniel',
 'n04536866': 'a photo of violin, fiddle',
 'n02730930': 'a photo of apron',
 'n11939491': 'a photo of daisy',
 'n02487347': 'a photo of macaque',
 'n02102318': 'a photo of cocker spaniel, English cocker spaniel, cocker',
 'n07714990': 'a photo of broccoli',
 'n04612504': 'a photo of yawl',
 'n03998194': 'a photo of prayer rug, prayer mat',
 'n01770081': 'a photo of harvestman, daddy longlegs, Phalangium opilio',
 'n02134084': 'a photo of ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
 'n02927161': 'a photo of butcher shop, meat market',
 'n04069434': 'a photo of reflex camera',
 'n03787032': 'a photo of mortarboard',
 'n02111889': 'a photo of Samoyed, Samoyede',
 'n03793489': 'a photo of mouse, computer mouse',
 'n07875152': 'a photo of potpie',
 'n02009229': 'a photo of little blue heron, Egretta caerulea',
 'n02086240': 'a photo of Shih-Tzu',
 'n07715103': 'a photo of cauliflower',
 'n02676566': 'a photo of acoustic guitar',
 'n04443257': 'a photo of tobacco shop, tobacconist shop, tobacconist',
 'n04483307': 'a photo of trimaran',
 'n03630383': 'a photo of lab coat, laboratory coat',
 'n02326432': 'a photo of hare',
 'n02124075': 'a photo of Egyptian cat',
 'n02280649': 'a photo of cabbage butterfly',
 'n02361337': 'a photo of marmot',
 'n02692877': 'a photo of airship, dirigible',
 'n04557648': 'a photo of water bottle',
 'n12267677': 'a photo of acorn',
 'n02165456': 'a photo of ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
 'n02112137': 'a photo of chow, chow chow',
 'n01914609': 'a photo of sea anemone, anemone',
 'n01735189': 'a photo of garter snake, grass snake',
 'n01944390': 'a photo of snail',
 'n03498962': 'a photo of hatchet',
 'n04356056': 'a photo of sunglasses, dark glasses, shades',
 'n02089973': 'a photo of English foxhound',
 'n02123597': 'a photo of Siamese cat, Siamese',
 'n04254680': 'a photo of soccer ball',
 'n02111500': 'a photo of Great Pyrenees',
 'n03394916': 'a photo of French horn, horn',
 'n07745940': 'a photo of strawberry',
 'n03709823': 'a photo of mailbag, postbag',
 'n07614500': 'a photo of ice cream, icecream',
 'n01704323': 'a photo of triceratops',
 'n03792782': 'a photo of mountain bike, all-terrain bike, off-roader',
 'n02883205': 'a photo of bow tie, bow-tie, bowtie',
 'n02101006': 'a photo of Gordon setter',
 'n01644373': 'a photo of tree frog, tree-frog',
 'n04366367': 'a photo of suspension bridge',
 'n02879718': 'a photo of bow',
 'n12144580': 'a photo of corn',
 'n07613480': 'a photo of trifle',
 'n04371430': 'a photo of swimming trunks, bathing trunks',
 'n03534580': 'a photo of hoopskirt, crinoline',
 'n03450230': 'a photo of gown',
 'n03938244': 'a photo of pillow',
 'n01443537': 'a photo of goldfish, Carassius auratus',
 'n02086079': 'a photo of Pekinese, Pekingese, Peke',
 'n02391049': 'a photo of zebra',
 'n02398521': 'a photo of hippopotamus, hippo, river horse, Hippopotamus amphibius',
 'n02093428': 'a photo of American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
 'n02107142': 'a photo of Doberman, Doberman pinscher',
 'n03208938': 'a photo of disk brake, disc brake',
 'n04399382': 'a photo of teddy, teddy bear',
 'n04118538': 'a photo of rugby ball',
 'n04355933': 'a photo of sunglass',
 'n04330267': 'a photo of stove',
 'n03045698': 'a photo of cloak',
 'n02128385': 'a photo of leopard, Panthera pardus',
 'n04442312': 'a photo of toaster',
 'n02356798': 'a photo of fox squirrel, eastern fox squirrel, Sciurus niger',
 'n03970156': "a photo of plunger, plumber's helper",
 'n02107683': 'a photo of Bernese mountain dog',
 'n03982430': 'a photo of pool table, billiard table, snooker table',
 'n02342885': 'a photo of hamster',
 'n09246464': 'a photo of cliff, drop, drop-off',
 'n02114367': 'a photo of timber wolf, grey wolf, gray wolf, Canis lupus',
 'n07747607': 'a photo of orange',
 'n02113712': 'a photo of miniature poodle',
 'n04487081': 'a photo of trolleybus, trolley coach, trackless trolley',
 'n04238763': 'a photo of slide rule, slipstick',
 'n02088466': 'a photo of bloodhound, sleuthhound',
 'n01630670': 'a photo of common newt, Triturus vulgaris',
 'n02117135': 'a photo of hyena, hyaena',
 'n04501370': 'a photo of turnstile',
 'n02098413': 'a photo of Lhasa, Lhasa apso',
 'n02127052': 'a photo of lynx, catamount',
 'n02782093': 'a photo of balloon',
 'n04548362': 'a photo of wallet, billfold, notecase, pocketbook',
 'n01496331': 'a photo of electric ray, crampfish, numbfish, torpedo',
 'n02841315': 'a photo of binoculars, field glasses, opera glasses',
 'n02480495': 'a photo of orangutan, orang, orangutang, Pongo pygmaeus',
 'n03961711': 'a photo of plate rack',
 'n02106030': 'a photo of collie',
 'n02397096': 'a photo of warthog',
 'n04041544': 'a photo of radio, wireless',
 'n02865351': 'a photo of bolo tie, bolo, bola tie, bola',
 'n03297495': 'a photo of espresso maker',
 'n01742172': 'a photo of boa constrictor, Constrictor constrictor',
 'n04252077': 'a photo of snowmobile',
 'n02087046': 'a photo of toy terrier',
 'n07768694': 'a photo of pomegranate',
 'n01582220': 'a photo of magpie',
 'n01930112': 'a photo of nematode, nematode worm, roundworm',
 'n01558993': 'a photo of robin, American robin, Turdus migratorius',
 'n02442845': 'a photo of mink',
 'n02105641': 'a photo of Old English sheepdog, bobtail',
 'n02120079': 'a photo of Arctic fox, white fox, Alopex lagopus',
 'n02085936': 'a photo of Maltese dog, Maltese terrier, Maltese',
 'n01694178': 'a photo of African chameleon, Chamaeleo chamaeleon',
 'n04371774': 'a photo of swing',
 'n03447721': 'a photo of gong, tam-tam',
 'n04251144': 'a photo of snorkel',
 'n01484850': 'a photo of great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
 'n02007558': 'a photo of flamingo',
 'n07753113': 'a photo of fig',
 'n02277742': 'a photo of ringlet, ringlet butterfly',
 'n03720891': 'a photo of maraca',
 'n02051845': 'a photo of pelican',
 'n02090622': 'a photo of borzoi, Russian wolfhound',
 'n02091831': 'a photo of Saluki, gazelle hound',
 'n02843684': 'a photo of birdhouse',
 'n04357314': 'a photo of sunscreen, sunblock, sun blocker',
 'n02437616': 'a photo of llama',
 'n02108422': 'a photo of bull mastiff',
 'n03729826': 'a photo of matchstick',
 'n02870880': 'a photo of bookcase'}
'''
