import os
from typing import Tuple, Any, Callable, List, Optional, Union

import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

from src.dataset import CustomDataset
from src.transform import TransformSelector
from models.models import ModelSelector

from config import config

# 모델 추론을 위한 함수
def inference(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader
):
    # 모델을 평가 모드로 설정
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            # 데이터를 같은 장치로 이동
            images = images.to(device)
            
            # 모델을 통해 예측 수행
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # 예측 결과 저장
            predictions.extend(preds.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가
    
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

# 추론에 사용할 Transform을 선언.
transform_selector = TransformSelector(transform_type = "sketch_albumentations")
test_transform = transform_selector.get_transform(is_train=False)

# 추론에 사용할 Dataset을 선언.
test_dataset = CustomDataset(root_dir=testdata_dir, info_df=test_info, transform=test_transform, is_inference=True)

# 추론에 사용할 DataLoader를 선언.
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,drop_last=False)

# 추론에 사용할 장비를 선택.
device = config.DEVICE if torch.cuda.is_available() else 'cpu'

# 추론에 사용할 Model을 선언.
model_selector = ModelSelector(emodel_type=config.MODEL_TYPE, num_classes=config.NUM_CLASSES, model_name=config.MODEL_NAME, pretrained=False)
model = model_selector.get_model()

# best epoch 모델을 불러오기.
model.load_state_dict(
    torch.load(
        os.path.join(save_result_path, "best_model.pt"),
        map_location='cpu'
    )
)

# predictions를 CSV에 저장할 때 형식을 맞춰서 저장
# 테스트 함수 호출
predictions = inference(model=model, device=config.DEVICE, test_loader=test_loader)

# 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
test_info['target'] = predictions
test_info = test_info.reset_index().rename(columns={"index": "ID"})

# DataFrame 저장
test_info.to_csv(f"{config.RESULT_DIR}/{model_name}_{config.EPOCHS}_output.csv", index=False)