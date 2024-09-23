# 필요 library들을 import합니다.
import os
import shutil
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
os.chdir('/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/suyoung')

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


# 새롭게 생성한 canny edge data를 train_df에 해당되는 것만 이동
source_dir = '/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/suyoung/data/canny'  # 원본 디렉토리 경로
destination_dir = '/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/suyoung/data/train'  # 대상 디렉토리 경로

source_folders = os.listdir(source_dir)
destination_folders = os.listdir(destination_dir)

for i in range(len(destination_folders)):
    if destination_folders[i] in source_folders:
        folder = destination_folders[i]
        image_path_list = train_df[train_df['class_name'] == folder]['image_path'].to_list()
        for j in range(len(image_path_list)):
            source_file_path = os.path.join(source_dir, image_path_list[j])
            file_name = image_path_list[j].split('/')[-1]
            new_name = 'edge_detected_' + file_name
            destination_file_path = os.path.join(destination_dir, folder, new_name)
            shutil.copy2(source_file_path, destination_file_path)

# train_info에 추가된 canny edge data를 덧붙임
class_name_list = []
image_path_list = []
target_list = []

traindata_folders = os.listdir(traindata_dir)

for i in range(len(traindata_folders)):
    if traindata_folders[i] != '.DS_Store':
        class_name = traindata_folders[i]
        class_path = os.path.join(traindata_dir, class_name)
        class_target = train_info[train_info['class_name'] == class_name]['target'].unique().item()

        # 클래스 폴더에서 이미지별로 path 추출
        class_images_list = os.listdir(class_path)
        for j in range(len(class_images_list)):
            class_image = class_images_list[j]
            if class_image.startswith('edge_detected'):
                class_image_path = os.path.join(class_name, class_image)
                class_name_list.append(class_name)
                image_path_list.append(class_image_path)
                target_list.append(class_target)

# 데이터 프레임에 추가하기 
new_data = pd.DataFrame({train_info.columns[0] : class_name_list, 
                         train_info.columns[1] : image_path_list,
                         train_info.columns[2] : target_list})

train_info = pd.concat([train_info, new_data], ignore_index=True)
train_info = train_info.astype({'target' : 'int'})


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

# # 학습에 사용할 Model을 선언.
# model_selector = ModelSelector(
#     model_type=config.MODEL_TYPE, 
#     num_classes=config.NUM_CLASSES,
#     model_name=config.MODEL_NAME, 
#     pretrained=config.PRETRAINED
# )

# model = model_selector.get_model()
# model.to(device)

# # 학습에 사용할 optimizer를 선언하고, learning rate를 지정
# optimizer = get_optimizer(model.parameters())
# scheduler = get_scheduler(config.SCHEDULER, optimizer, steps_per_epoch=len(train_loader))


# # 학습에 사용할 Loss를 선언.
# loss_fn = get_loss_function()

# 앞서 선언한 필요 class와 변수들을 조합해, 학습을 진행할 Trainer를 선언. 
trainer = get_trainer(train_loader, val_loader) 
trainer.train()