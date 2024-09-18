# 필요 library들을 import합니다.
import os
from typing import Tuple, Any, Callable, List, Optional, Union

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from src.transform import *
from src.dataset import *
from src.trainer import *
from models.models import *
from models.loss import *

from config import config

# 학습에 사용할 장비를 선택.
# torch라이브러리에서 gpu를 인식할 경우, cuda로 설정.
device = config.DEVICE

# 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정.
model_name = config.MODEL_NAME
traindata_dir = config.TRAIN_DATA_DIR
traindata_info_file = os.path.join(traindata_dir, '../train.csv')
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
    stratify=train_info['target']
)

# 학습에 사용할 Transform을 선언.
transform_selector = TransformSelector(
    transform_type = "albumentations"
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

# 학습에 사용할 Model을 선언.
model_selector = ModelSelector(
    model_type=config.MODEL_TYPE, 
    num_classes=config.NUM_CLASSES,
    model_name=config.MODEL_NAME, 
    pretrained=config.PRETRAINED
)

model = model_selector.get_model()
model.to(device)

# 학습에 사용할 optimizer를 선언하고, learning rate를 지정
optimizer = get_optimizer(model.parameters())
scheduler = get_scheduler(config.SCHEDULER, optimizer, steps_per_epoch=len(train_loader))


# 학습에 사용할 Loss를 선언.
loss_fn = get_loss_function()

# 앞서 선언한 필요 class와 변수들을 조합해, 학습을 진행할 Trainer를 선언. 
trainer = Trainer(
    model=model, 
    device=device, 
    train_loader=train_loader,
    val_loader=val_loader, 
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn, 
    epochs=config.EPOCHS,
    result_path=save_result_path
)

trainer.train()