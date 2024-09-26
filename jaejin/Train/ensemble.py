import os
from typing import Tuple, Any, Callable, List, Optional, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

#from src.transform import *
#from models.models import *
from functions import *

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

testdata_dir = '/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/data/test'
testdata_info_file = os.path.join(testdata_dir, '../test.csv')
testdata_info_file = os.path.abspath(testdata_info_file)
#save_result_path = f"./train_result/{MODEL_NAME}_{LEARNING_RATE}"

# if not os.path.exists(save_result_path):
#     os.makedirs(save_result_path)
print(testdata_info_file)
test_info = pd.read_csv(testdata_info_file)
num_classes = 500

transform_selector = TransformSelector(transform_type="albumentations")
test_transform = transform_selector.get_transform(is_train=True)

test_dataset = CustomDataset(
    root_dir=testdata_dir,
    info_df=test_info,
    transform=test_transform,
    is_inference=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=64,
    shuffle=False,
    drop_last=False
)

device = "cuda" if torch.cuda.is_available() else 'cpu'

# 앙상블할 모델 목록 정의
model_configs = [
    {"model_type": 'timm', "model_name": "efficientnet_b1", "weights_path": "efficientnet_b1_0.0003_aug_False_1_Acc_0.7832_best_model.pt"},
    {"model_type": 'timm', "model_name": "convnext_base", "weights_path": "conv.pt"},
   {"model_type": 'timm', "model_name": "dino-vitb8", "weights_path": "dino-vitb8_0.0008_aug_False1_Acc_0.7464_best_model.pt"},
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
        num_classes=500,
        model_name=model_configs[1]["model_name"],
        pretrained=False
    )
model = model_selector.get_model()
model.load_state_dict(torch.load(model_configs[1]["weights_path"], map_location='cpu'))


modelss.append(model)


model,_ = get_model_and_transforms(model_configs[2]["model_name"],val="1")
model.load_state_dict(
    torch.load(
        os.path.join(dir, model_configs[2]["weights_path"]),
        map_location='cpu'
    )
)
modelss.append(model)


# 앙상블 추론 실행
ensemble_predictions = ensemble_inference(modelss, device, test_loader)

from clipeval import *
clip = getClip()
t = clip.inference()

keys_list = list(get_imagenet_ditction(mini=True,values=False).keys())
tt=[keys_list[index] for index in t]
name_info_file = dir+"/data/name_label.csv"
name_data = pd.read_csv(name_info_file)
test_info = np.array([name_data[name_data["class_name"] == classes]["target"].values for classes in tt])
a=np.argmax(ensemble_predictions[0],axis=1)
b=np.argmax(ensemble_predictions[1],axis=1)
c=np.argmax(ensemble_predictions[2],axis=1)
d = test_info.squeeze()
traindata_info_file = "/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/data/train.csv"
train_info = pd.read_csv(traindata_info_file)
e = np.array(train_info["target"])
count= [0,0,0,0]
for index in range(len(e)):
    if a[index] == e[index]:
            count[0]+=1
    if b[index] == e[index]:
            count[1]+=1
    if c[index] == e[index]:
            count[2]+=1
    if d[index] == e[index]:
            count[3]+=1

for co in count:
    print(co/len(e))

balanced_dataset_info = []
for index in range(len(e)):  
  
    balanced_dataset_info.append({
        'effi': a[index],
        'conv': b[index],
        'dino': c[index],
        'clip': d[index],
        'tar': e[index]
    })
pd.DataFrame(balanced_dataset_info).to_csv('tempVal.csv', index=False)