import os
import sys
import cv2
from typing import Tuple, Any, Callable, List, Optional, Union
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from functions import TransformSelector
from tqdm.auto import tqdm
import pandas as pd
import wandb


from sklearn.model_selection import train_test_split

dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
traindata_dir = dir+"/data/train"
traindata_info_file = dir+"/data/train.csv"
save_result_path = dir+"/train_result"
lowest_val_loss = float('inf')
val_losses = []
best_models = []

class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable,
        is_inference: bool = False
    ):
        
        self.root_dir = root_dir  
        self.transform = transform  
        self.is_inference = is_inference 
        self.image_paths = info_df['image_path'].tolist()  
        if not self.is_inference:
            self.targets = info_df['target'].tolist()  

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])  
        image = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = self.transform(image=image)['image'] 
        if self.is_inference:
            return image
        else:
            target = self.targets[index]  # 해당 이미지의 레이블
            return image, target  # 변환된 이미지와 레이블을 튜플 형태로 반환합니다. 
        
def save_model(epoch, loss, model_name, model):
    global dir
    global traindata_dir
    global traindata_info_file
    global save_result_path
    global lowest_val_loss
    global val_losses
    global best_models
    # 모델 저장 경로 설정
    os.makedirs(save_result_path+"/"+model_name, exist_ok=True)

    # 현재 에폭 모델 저장
    current_model_path = os.path.join(save_result_path+"/"+model_name, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
    torch.save(model.state_dict(), current_model_path)

    # 최상위 3개 모델 관리
    best_models.append((loss, epoch, current_model_path))
    best_models.sort()
    if len(best_models) > 2:
        _, _, path_to_remove = best_models.pop(-1)  # 가장 높은 손실 모델 삭제
        if os.path.exists(path_to_remove):
            os.remove(path_to_remove)

    # 가장 낮은 손실의 모델 저장
    if loss < lowest_val_loss:
        lowest_val_loss = loss
        best_model_path = os.path.join(save_result_path+"/"+model_name, model_name+'.pt')
        torch.save(model.state_dict(), best_model_path)
        print(f"Save {epoch}epoch result. Loss = {loss:.4f}")

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

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        preprocess = weights.transforms()

    elif model_name == "vit_l_16":
        weights = models.ViT_L_16_Weights.DEFAULT
        model = models.vit_l_16(weights=weights)
        preprocess = weights.transforms()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model, preprocess

def evaluate_model(model, data_loader,criterion, device):
    progress_bar = tqdm(data_loader, desc="Validating", leave=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())

    return correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        val_accuracy = evaluate_model(model, val_loader, criterion, device)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy
        })
        save_model(epoch, loss, model_name, model)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

def main(model_name):

    wandb.init(project="model_comparison", name=model_name)

    train_info = pd.read_csv(traindata_info_file)
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        random_state=20,
        stratify=train_info['target']
        )
    
    num_classes = len(train_info['target'].unique())

    transform_selector = TransformSelector(
        transform_type = "albumentations"
        )
    

    data_transforms = A.Compose([
        A.Resize(224, 224), 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ToTensorV2() 
    ])  

    train_dataset = CustomDataset(
    root_dir=traindata_dir,
    info_df=train_df,
    transform=data_transforms)

    val_dataset = CustomDataset(
    root_dir=traindata_dir,
    info_df=val_df,
    transform=data_transforms
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=128, 
        shuffle=False
    )    
      

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining and evaluating {model_name}")
    model, _ = get_model_and_transforms(model_name)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    pretrained_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f"Pretrained {model_name} accuracy: {pretrained_accuracy:.4f}")
    wandb.log({f"{model_name}_pretrained_accuracy": pretrained_accuracy})
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    
    final_accuracy = evaluate_model(model, val_loader, device)
    print(f"Final {model_name} accuracy: {final_accuracy:.4f}")
    wandb.log({f"{model_name}_final_accuracy": final_accuracy})

    wandb.finish()
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_models.py <model_name> <model_version>")
        sys.exit(1)
    model_name = sys.argv[1]
    main(model_name)