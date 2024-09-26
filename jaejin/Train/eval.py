import os
import sys
import cv2
from typing import Tuple, Any, Callable, List, Optional, Union
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from functions import CustomDataset, Trainer, get_model_and_transforms
from sklearn.model_selection import train_test_split

def main(model_name,model_rl,is_ag):
    dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
    traindata_dir = dir+"/data/train"
    traindata_info_file = dir+"/data/train.csv"
    save_result_path = dir+"/train_result"

    train_info = pd.read_csv(traindata_info_file)
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        random_state=20,
        stratify=train_info['target']
        )
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining and evaluating {model_name}")
    model, preprocess = get_model_and_transforms(model_name)
    model = model.to(device)

    if is_ag:
        data_transforms = A.Compose([
        # Geometric transformations
            A.Rotate(limit=10, p=0.5),
            A.Affine(scale=(0.8, 1.2), shear=(-10, 10), p=0.5),
            A.ElasticTransform(alpha=1, sigma=10, p=0.5),
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
            A.Resize(224, 224), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2() 
        ])  
    else:
        data_transforms = A.Compose([
            A.Resize(224, 224), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2() 
        ])  

    val_data_transforms = A.Compose([
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
    transform=val_data_transforms
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False
    )    

    # 스케줄러 초기화
    scheduler_step_size = 30 # 매 30step마다 학습률 감소
    scheduler_gamma = 1  

    # 한 epoch당 step 수 계산
    steps_per_epoch = len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr= model_rl)
    loss_fn = nn.CrossEntropyLoss()
    # 2 epoch마다 학습률을 감소시키는 스케줄러 선언
    epochs_per_lr_decay = 5
    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=scheduler_step_size, 
        gamma=scheduler_gamma
    )    


    trainer = Trainer(model,
            device,
            train_loader, 
            val_loader,
            optimizer, 
            scheduler, 
            loss_fn,  
            epochs=50, 
            result_path=save_result_path,
            model_name=model_name+'_'+str(model_rl)+'_aug_'+str(is_ag))   
    
    trainer.train()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_models.py <model_name> <model_rl>")
        sys.exit(1)
    model_name = sys.argv[1]
    model_rl = sys.argv[2]
    is_ag = sys.argv[3] == 'True'
    main(model_name,float(model_rl),is_ag)
    