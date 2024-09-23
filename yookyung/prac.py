import numpy as np
import torch


def cutmix(image, label, dataset, num_classes, alpha=1.0):
    # 배치 크기 설정 (1로 고정)
    batch_size = 1
    lam = np.random.beta(alpha, alpha)

    # 랜덤으로 다른 이미지와 레이블 선택
    rand_index = torch.randint(len(dataset), (1,)).item()
    rand_image, rand_label = dataset[rand_index]

    # 배치 차원을 추가
    image = image.unsqueeze(0)  # (1, C, H, W)
    rand_image = rand_image.unsqueeze(0)  # (1, C, H, W)

    # 이미지를 결합
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = rand_image[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))

    # 소프트 레이블 생성
    soft_label = torch.zeros(batch_size, num_classes)
    soft_label.scatter_(1, label.unsqueeze(0).long(), lam)
    soft_label.scatter_(1, rand_label.unsqueeze(0).long(), 1 - lam)

    return image.squeeze(0), soft_label  # 원래 차원으로 되돌림

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
