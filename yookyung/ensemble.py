import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from config import config

# 모델 로드 함수
def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Soft Voting 앙상블 함수
def soft_voting_ensemble(models, dataloader, device):
    soft_predictions = []

    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)
            logits_sum = torch.zeros((images.shape[0], models[0].num_classes), device=device)
            
            # 각 모델의 예측 결과를 확률 형태로 더해줌
            for model in models:
                logits = model(images)
                logits_sum += F.softmax(logits, dim=1)  # softmax로 확률을 구한 후 더함
            
            # 확률 평균을 구한 뒤 최종 클래스 예측
            avg_logits = logits_sum / len(models)
            soft_predictions.append(avg_logits.cpu().numpy())

    return np.vstack(soft_predictions)

# Hard Voting 앙상블 함수
def hard_voting_ensemble(models, dataloader, device):
    hard_predictions = []

    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)
            votes = np.zeros((images.shape[0], len(models)))

            # 각 모델의 예측 클래스 저장
            for i, model in enumerate(models):
                logits = model(images)
                preds = logits.argmax(dim=1)  # 각 모델의 예측 클래스
                votes[:, i] = preds.cpu().numpy()
            
            # 다수결을 통해 최종 클래스 결정
            final_preds = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=votes)
            hard_predictions.append(final_preds)

    return np.concatenate(hard_predictions)

# 예시 실행 코드
def ensemble_predict(model_class, model_paths, dataloader, device, ensemble_type="soft"):
    """
    model_class: 모델 클래스 정의
    model_paths: 모델 가중치 파일 경로 리스트
    dataloader: 데이터 로더
    device: cuda 또는 cpu
    ensemble_type: 'soft' 또는 'hard'
    """
    # 모델을 불러오기
    models = [load_model(model_class(), path, device) for path in model_paths]

    if ensemble_type == "soft":
        return soft_voting_ensemble(models, dataloader, device)
    elif ensemble_type == "hard":
        return hard_voting_ensemble(models, dataloader, device)
    else:
        raise ValueError("Unsupported ensemble type: choose 'soft' or 'hard'.")

model_paths = []
model_class = []
device = config.DEVICE if torch.cuda.is_available() else 'cpu'

# # soft voting 앙상블
# soft_preds = ensemble_predict(model_class=DenseNet121, model_paths=model_paths, dataloader=test_loader, device=device, ensemble_type="soft")

# # hard voting 앙상블
# hard_preds = ensemble_predict(model_class=DenseNet121, model_paths=model_paths, dataloader=test_loader, device=device, ensemble_type="hard")






import os
from typing import Tuple, Any, Callable, List, Optional, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

from src.dataset import *
from src.transform import *
from models.models import *

from config import config

def inference(model: nn.Module, device: torch.device, test_loader: DataLoader):
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs.cpu().detach().numpy())
    
    return np.concatenate(predictions)

def ensemble_inference(models: List[nn.Module], device: torch.device, test_loader: DataLoader):
    ensemble_predictions = []
    
    for model in models:
        model_predictions = inference(model, device, test_loader)
        ensemble_predictions.append(model_predictions)
    
    # 모든 모델의 예측을 평균내어 앙상블 수행
    ensemble_result = np.mean(ensemble_predictions, axis=0)

    return ensemble_result

testdata_dir = config.TEST_DATA_DIR
testdata_info_file = os.path.join(testdata_dir, '../test.csv')
testdata_info_file = os.path.abspath(testdata_info_file)
save_result_path = config.CHECKPOINT_DIR

if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

test_info = pd.read_csv(testdata_info_file)
num_classes = config.NUM_CLASSES

transform_selector = TransformSelector(transform_type="albumentations")
test_transform = transform_selector.get_transform(is_train=False)

test_dataset = CustomDataset(
    root_dir=testdata_dir,
    info_df=test_info,
    transform=test_transform,
    is_inference=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=config.BATCH_SIZE, 
    shuffle=False,
    drop_last=False
)

device = config.DEVICE if torch.cuda.is_available() else 'cpu'

# 앙상블할 모델 목록 정의
model_configs = [
    {"model_type": 'timm', "model_name": "convnext_base", "weights_path": "../train_result/convnext_base/best_model.pt"},
    {"model_type": 'timm', "model_name": "convnext_base", "weights_path": "../train_result/convnext_base_30epoch_0.8860/best_model.pt"},
]

# 모델 로드 및 앙상블 수행
models = []
for model_config in model_configs:
    model_selector = ModelSelector(
        model_type=model_config["model_type"],
        num_classes=config.NUM_CLASSES,
        model_name=model_config["model_name"],
        pretrained=False
    )
    model = model_selector.get_model()
    model.load_state_dict(torch.load(model_config["weights_path"], map_location='cpu'))
    models.append(model)

# 앙상블 추론 실행
ensemble_predictions = ensemble_inference(models, device, test_loader)

# 앙상블 결과 처리 및 저장
ensemble_classes = np.argmax(ensemble_predictions, axis=1)
test_info['target'] = ensemble_classes
test_info = test_info.reset_index().rename(columns={"index": "ID"})

# 결과 저장
ensemble_output_path = f"{config.RESULT_DIR}/ensemble_output.csv"
test_info.to_csv(ensemble_output_path, index=False)

print(f"Ensemble predictions saved to {ensemble_output_path}")