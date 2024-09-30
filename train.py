import os
from typing import Tuple, Any, Callable, List, Optional, Union

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from src.transform import TransformSelector
from src.dataset import CustomDataset
from src.trainer import DataLoader, get_trainer

from config import config


device = config.DEVICE if torch.cuda.is_available() else 'cpu'
model_name = config.MODEL_NAME

if not config.NEW_DATASET: # 원본 데이터셋 사용할 경우
    traindata_dir = config.TRAIN_DATA_DIR
    valdata_dir = traindata_dir

    traindata_info_file = os.path.join(traindata_dir, '../train.csv')
    traindata_info_file = os.path.abspath(traindata_info_file)

    train_info = pd.read_csv(traindata_info_file)
    train_df, val_df = train_test_split(train_info, test_size=1-config.TRAIN_RATIO, stratify=train_info['target'], random_state=20)

else: # 새로운 데이터셋 사용할 경우 
    traindata_dir = "./data/trainDelFlip_objectsplit_train_upWithcanny"
    valdata_dir = "./data/trainDelFlip_objectsplit_val"

    train_df = pd.read_csv("./data/trainDelFlip_objectsplit_train_upWithcanny.csv")
    val_df = pd.read_csv("./data/trainDelFlip_objectsplit_val.csv")

save_result_path = config.CHECKPOINT_DIR
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

# 학습에 사용할 Transform을 선언.
transform_selector = TransformSelector(transform_type = 'sketch_albumentations')
train_transform = transform_selector.get_transform(is_train=True)
val_transform = transform_selector.get_transform(is_train=False)

# 학습에 사용할 Dataset을 선언.
train_dataset = CustomDataset(root_dir=traindata_dir, info_df=train_df, transform=train_transform)
val_dataset = CustomDataset(root_dir=valdata_dir, info_df=val_df, transform=val_transform)

# 학습에 사용할 DataLoader를 선언.
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

trainer = get_trainer(train_loader, val_loader) 
trainer.train()