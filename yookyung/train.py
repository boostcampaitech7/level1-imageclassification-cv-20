# 필요 library들을 import합니다.
import os
from typing import Tuple, Any, Callable, List, Optional, Union

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from src.transform import TransformSelector
from src.dataset import CustomDataset
from src.trainer import DataLoader, get_trainer

from config import config

# 학습에 사용할 장비를 선택.
# torch라이브러리에서 gpu를 인식할 경우, cuda로 설정.
device = config.DEVICE if torch.cuda.is_available() else 'cpu'

# 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정.
model_name = config.MODEL_NAME
traindata_dir = config.TRAIN_DATA_DIR
traindata_info_file = os.path.join(traindata_dir, '../train.csv')
traindata_info_file = os.path.abspath(traindata_info_file)
save_result_path = config.CHECKPOINT_DIR

if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

# 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
train_info = pd.read_csv(traindata_info_file)
num_classes = config.NUM_CLASSES

# 각 class별로 8:2의 비율이 되도록 학습과 검증 데이터를 분리.
train_df, val_df = train_test_split(
    train_info, 
    test_size=1-config.TRAIN_RATIO,
    stratify=train_info['target'],
    random_state=20
)

# 학습에 사용할 Transform을 선언.
transform_selector = TransformSelector(
    transform_type = 'sketch_albumentations'
)
train_transform = transform_selector.get_transform(is_train=True)
val_transform = transform_selector.get_transform(is_train=False)

# 학습에 사용할 Dataset을 선언.
train_dataset = CustomDataset(
    root_dir=traindata_dir,
    info_df=train_df,
    transform=train_transform
)
val_dataset = CustomDataset(
    root_dir=traindata_dir,
    info_df=val_df,
    transform=val_transform
)

# 학습에 사용할 DataLoader를 선언.
train_loader = DataLoader(
    train_dataset, 
    batch_size=config.BATCH_SIZE, 
    shuffle=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=config.BATCH_SIZE, 
    shuffle=False
)

trainer = get_trainer(train_loader, val_loader) 
trainer.train()