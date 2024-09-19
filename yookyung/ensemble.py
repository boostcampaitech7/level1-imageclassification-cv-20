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

# soft voting 앙상블
soft_preds = ensemble_predict(model_class=DenseNet121, model_paths=model_paths, dataloader=test_loader, device=device, ensemble_type="soft")

# hard voting 앙상블
hard_preds = ensemble_predict(model_class=DenseNet121, model_paths=model_paths, dataloader=test_loader, device=device, ensemble_type="hard")
