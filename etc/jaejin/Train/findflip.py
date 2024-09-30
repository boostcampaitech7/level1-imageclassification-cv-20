import os
import numpy as np
from skimage import io, transform, exposure
from skimage.feature import hog
from tqdm import tqdm
from skimage.color import rgb2gray, gray2rgb
from skimage.metrics import structural_similarity as ssim
import pandas as pd
def preprocess_image(image):
    if image.ndim == 2:
        image = gray2rgb(image)
    # 이미지 크기 표준화
    #image = transform.resize(image, (224, 224), anti_aliasing=True)
    # 비트 수준 표준화 (0-255 범위로 변환)
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = image.astype(np.uint8)
    return image

def has_similar_features(image1, image2):

    
    fd1, _ = hog(image1, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    fd2, _ = hog(image2, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    
    hog_similarity = np.dot(fd1, fd2) / (np.linalg.norm(fd1) * np.linalg.norm(fd2))
    return hog_similarity > 0.90
        
    # # return similarity > 0.50  # 유사도 임계값 설정
    # ssim_score = ssim(image1, image2, 
    #               data_range=image1.max() - image2.min(),
    #               multichannel=True,  # 다중 채널 이미지
    #               win_size=7,  # 윈도우 크기 명시적 지정
    #               channel_axis=2)  
    
    # # HOG와 SSIM의 가중 평균
    # combined_similarity = 0.7 * hog_similarity + 0.3 * ssim_score
    # if combined_similarity > 0.85:
    #     print(combined_similarity,hog_similarity,ssim_score)
    # return combined_similarity > 0.9

def is_flipped_image(image_path1, image_path2):
    try:
        image1 = io.imread(image_path1)
        image2 = io.imread(image_path2)
    except Exception as e:
        print(f"Error loading images: {e}")
        return False, None

    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    if image1.shape != image2.shape:
        return False, "none"
    
    # 원본이랑 비교
    if has_similar_features(image1, image2):
        return True, "origin"

    # 좌우 대칭 비교
    flipped_lr = np.fliplr(image2)
    if has_similar_features(image1, flipped_lr):
        return True, "horizontal"
    # 상하 대칭 비교
    flipped_ud = np.flipud(image2)
    if has_similar_features(image1, flipped_ud):
        return True, "vertical"
    # 원점
    flipped_sim = np.flipud(flipped_lr)
    if has_similar_features(image1, flipped_sim):
        return True, "symmetry"
    
    return False, "none"

def find_flipped_images(data_dir):
    flipped_pairs = {}
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for class_name in tqdm(class_names,desc="total progress"):
        class_dir = os.path.join(data_dir, class_name)        
        if not os.path.isdir(class_dir):
            continue        
        image_files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))])        
        flipped_pairs[class_name] = []
        selected = [ False for i in range(len(image_files))]
        for i in range(len(image_files)):
            if not selected[i]:
                for j in range(i+1, len(image_files)):
                    if not selected[j]:
                        image_path1 = os.path.join(class_dir, image_files[i])
                        image_path2 = os.path.join(class_dir, image_files[j])   
                        is_flipped, flip_type = is_flipped_image(image_path1, image_path2)
                        if is_flipped:
                            flipped_pairs[class_name].append((image_path1, image_path2))
                            selected[j] = True
                            data.append({
                                'class_name': class_name, 
                                'original_filename': image_path1.split('/')[-1], 
                                'flipped_filename': image_path2.split('/')[-1],
                                'type': flip_type
                            })
        
        df = pd.DataFrame(data)
        output_file = 'flipped_images_info.csv'
        df.to_csv(output_file, index=False)
        remove_flipped_images(flipped_pairs[class_name])

    return flipped_pairs

def print_flipped_images_summary(flipped_pairs):
    print("\nSummary of flipped images by class:")
    for class_name, pairs in flipped_pairs.items():
        print(f"{class_name}: {len(pairs)} flipped image pairs")

def remove_flipped_images(flipped_pairs):
    for pair in flipped_pairs:
        # 두 번째 이미지를 삭제 (첫 번째 이미지 유지)
        try:
            os.remove(pair[1])
            # print(f"Removed flipped image: {pair[1]}")
        except Exception as e:
            print(f"Failed to remove image: {pair[1]} with error: {e}")

# 메인 실행 코드
data = []
data_directory = dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/data/trainDelFlip"
flipped_pairs = find_flipped_images(data_directory)
df = pd.DataFrame(data)

# CSV 파일로 저장
output_file = 'flipped_images_info.csv'
df.to_csv(output_file, index=False)
print(f"Found {len(flipped_pairs)} pairs of potentially flipped images.")