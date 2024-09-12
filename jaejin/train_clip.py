import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
from transformers import CLIPProcessor
from functions import *

import os


# 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정
traindata_dir = "../data/train"
traindata_info_file = "../data/train.csv"

# 테스트 데이터의 경로와 정보를 가진 파일의 경로를 설정
testdata_dir = "../data/test"
testdata_info_file = "../data/test.csv"

# 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기
train_data = pd.read_csv(traindata_info_file)

# 테스트 데이터
test_data = pd.read_csv(testdata_info_file)

# 학습 데이터의 정보를 출력
train_info = train_data.info()
train_head = train_data.head()

train_info, train_head

# 데이터의 기본적인 통계 정보를 출력
data_description = train_data.describe(include='all')

# class_name의 unique한 값의 개수를 출력
unique_classes = train_data['class_name'].nunique()

# target의 unique한 값의 개수를 출력
unique_targets = train_data['target'].nunique()

data_description, unique_classes, unique_targets

from transformers import CLIPProcessor, CLIPModel,AdamW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
train_info = pd.read_csv(traindata_info_file)

# 총 class의 수를 측정.
num_classes = len(train_info['target'].unique())

# 각 class별로 8:2의 비율이 되도록 학습과 검증 데이터를 분리.
train_df, val_df = train_test_split(
    train_info, 
    test_size=0.2,
    stratify=train_info['target']
)

# 학습에 사용할 Transform을 선언.
transform_selector = TransformSelector(
    transform_type = "AlbumentationsTransformTest"
)
train_transform = transform_selector.get_transform(is_train=True)
val_transform = transform_selector.get_transform(is_train=False)

train_dataset = CLIPDataset(
    root_dir=traindata_dir,
    info_df=train_df,
    transform=train_transform,
    processor=processor,
    device=device,
    use_print = False
)
val_dataset = CLIPDataset(
    root_dir=traindata_dir,
    info_df=val_df,
    transform=val_transform,
    processor=processor,
    device=device,
    use_print = False
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=64, 
    shuffle=False
)

# 스케줄러 초기화
scheduler_step_size = 30 # 매 30step마다 학습률 감소
scheduler_gamma = 0.99  # 학습률을 현재의 10%로 감소

# 한 epoch당 step 수 계산
steps_per_epoch = len(train_loader)
print(steps_per_epoch)
optimizer = AdamW(model.parameters(), lr=5e-6)

# 2 epoch마다 학습률을 감소시키는 스케줄러 선언
epochs_per_lr_decay = 5
scheduler_step_size = steps_per_epoch * epochs_per_lr_decay

scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=scheduler_step_size, 
    gamma=scheduler_gamma
)

loss_fn = Loss()

save_result_path = "./train_result"
mini_values=get_imagenet_ditction(mini=True,values=True)
trainer = CLIP_Trainer(
    model=model.to(device), 
    device=device, 
    train_loader=train_loader,
    val_loader=val_loader, 
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn, 
    processor=processor,
    epochs=100,
    result_path=save_result_path,
    mini_values=mini_values
)

# 모델 학습.
trainer.train()