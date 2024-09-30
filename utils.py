import numpy as np
import torch

def cutmix_batch(images, labels, alpha=1.0):
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)
    
    rand_index = torch.randperm(batch_size)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    
    new_images = images.clone()
    new_images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    new_labels = lam * labels + (1 - lam) * labels[rand_index]
    
    return new_images, new_labels

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