import os
from typing import Tuple, Any, Callable, List, Optional, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

from src.dataset import *
#from src.transform import *
#from models.models import *
from functions import *
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
    #ensemble_result = np.mean(ensemble_predictions, axis=0)

    return ensemble_predictions

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
    {"model_type": 'timm', "model_name": "efficientnet_b1", "weights_path": "effi.pt"},
    {"model_type": 'timm', "model_name": "convnext_base", "weights_path": "conv.pt"},
   {"model_type": 'timm', "model_name": "dino-vitb8", "weights_path": "dino.pt"},
]
dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/jaejin/Train"
modelss = []

model,_ = get_model_and_transforms(model_configs[0]["model_name"])
model.load_state_dict(
    torch.load(
        os.path.join(dir, model_configs[0]["weights_path"]),
        map_location='cpu'
    )
)
modelss.append(model)


model_selector = ModelSelector(
        model_type=model_configs[1]["model_type"],
        num_classes=config.NUM_CLASSES,
        model_name=model_configs[1]["model_name"],
        pretrained=False
    )
model = model_selector.get_model()
model.load_state_dict(torch.load(model_configs[1]["weights_path"], map_location='cpu'))


modelss.append(model)


model,_ = get_model_and_transforms(model_configs[2]["model_name"],val="2")
model.load_state_dict(
    torch.load(
        os.path.join(dir, model_configs[2]["weights_path"]),
        map_location='cpu'
    )
)
modelss.append(model)


# 앙상블 추론 실행
ensemble_predictions = ensemble_inference(modelss, device, test_loader)
print(ensemble_predictions[0].shape)
# # 앙상블 결과 처리 및 저장
# ensemble_classes = np.argmax(ensemble_predictions, axis=1)
# test_info['target'] = ensemble_classes
# test_info = test_info.reset_index().rename(columns={"index": "ID"})

# # 결과 저장
# ensemble_output_path = f"{config.RESULT_DIR}/ensemble_output.csv"
# test_info.to_csv(ensemble_output_path, index=False)

# print(f"Ensemble predictions saved to {ensemble_output_path}")