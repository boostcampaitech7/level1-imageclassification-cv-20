import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision import transforms
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm.auto import tqdm


#CutMix 함수 정의
def cutmix(image1, image2, alpha=1.0):
    image1 = image1.resize((224, 224))
    image2 = image2.resize((224, 224))
    image1 = np.array(image1)
    image2 = np.array(image2)
    if len(image1.shape) == 2:
        image1 = np.stack((image1,)*3, axis=-1)
    if len(image2.shape) == 2:
        image2 = np.stack((image2,)*3, axis=-1)
    
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image1.shape[:2], lam)
    
    image1[bbx1:bbx2, bby1:bby2] = image2[bbx1:bbx2, bby1:bby2]
    return Image.fromarray(image1.astype(np.uint8))

def rand_bbox(size, lam):
    W, H = size
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

def is_image_file(filename):
    # 이미지 파일 확장자 리스트
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def save_image_as_jpeg(img, save_path, quality=95):
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        img_pil = img
    else:
        raise TypeError("Unsupported image type")
    
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_pil.save(save_path, "JPEG", quality=quality)
# RandAugment 정의

# rand_augment = A.Compose([
#     A.RandomRotate90(),
#     A.Flip(),
#     A.Transpose(),
#     A.OneOf([
#         A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
#     ], p=0.2),
#     A.OneOf([
#         A.MotionBlur(p=0.2),
#         A.MedianBlur(blur_limit=3, p=0.1),
#         A.Blur(blur_limit=3, p=0.1),
#     ], p=0.2),
#     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
#     A.OneOf([
#         A.OpticalDistortion(p=0.3),
#         A.GridDistortion(p=0.1),
#     ], p=0.2),
#     ToTensorV2(),
# ])

def oversampling(data_dir, output_dir):
    # 클래스별 이미지 수 확인
    class_names = os.listdir(data_dir)
    class_distribution = {class_name: len(os.listdir(os.path.join(data_dir, class_name))) for class_name in class_names if class_name[0] == 'n'}

    # 목표 이미지 수 설정 (예: 가장 많은 클래스의 이미지 수)
    target_count = max(class_distribution.values())
    print(target_count)
    # 각 클래스에 대해 증강 수행
    for class_name, count in tqdm(class_distribution.items()):
        class_dir = os.path.join(data_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        images = [Image.open(os.path.join(class_dir, img)) for img in os.listdir(class_dir) 
            if is_image_file(img) and not img.startswith('.')]
        
        #images = np.array(images, dtype=object) 
        # 원본 이미지 복사
        for i, img in enumerate(images):
            new_array = np.array(img)
            new_array.resize((224, 224))
            save_image_as_jpeg(img, os.path.join(output_class_dir, f"{class_name}_{i}.JPEG"))


        # 부족한 만큼 이미지 증강
        augmented_count = count
        while augmented_count < target_count:
            # CutMix
            # if len(images) > 1:
            #     img1, img2 = random.sample(list(images), 2)
            #     new_img = cutmix(img1, img2)
            # else:
            #     new_img = images[0].copy().resize((224, 224))

            # cutmix 할거면 이곳 삭제
            img1= random.sample(list(images), 1)[0]
            img1 = img1.resize((224, 224))
            img1 = np.array(img1)
            if len(img1.shape) == 2:
                img1 = np.stack((img1,)*3, axis=-1)
            
            # RandAugment
            new_img = rand_augment(image=np.array(img1))['image']
            new_img = new_img.permute(1, 2, 0).byte().numpy()

            save_image_as_jpeg(new_img, os.path.join(output_class_dir, f"{class_name}_aug_{augmented_count}.JPEG"))
            augmented_count += 1

    print("Balanced dataset created and saved.")

# 데이터셋 경로
dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
data_dir = dir+'/data/trainDelFlip_objectsplit_train'
output_dir = dir+'/data/trainDelFlip_objectsplit_train_up'

rand_augment = A.Compose([
    A.HorizontalFlip(p=0.5),  # 좌우 반전
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),  # 밝기 및 대비 조정
    A.GaussNoise(var_limit=(30.0, 40.0), p=0.7),  # 약한 노이즈 추가
    A.ElasticTransform(alpha=2.0, sigma=50, alpha_affine=16, p=0.7),  # 약한 탄성 변형 추가
    A.CoarseDropout(max_holes=20, max_height=20, max_width=20, min_holes=1, fill_value=0, p=0.8),
    ToTensorV2()  # 텐서로 변환
])
dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
oversampling(data_dir = dir+'/data/trainDelFlip_objectsplit_train',
            output_dir = dir+'/data/trainDelFlip_objectsplit_train_up'
            )

