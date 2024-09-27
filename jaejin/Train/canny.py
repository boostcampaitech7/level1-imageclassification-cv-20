import os
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from glob import glob
import numpy as np # type: ignore
import cv2 # type: ignore
from tqdm.auto import tqdm
# 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정
dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
traindata_dir = dir+"/data/trainDelFlip_objectsplit_train_up"
traindata_info_file = dir+"/data/trainDelFlip_objectsplit_train_up.csv"


train_data = pd.read_csv(traindata_info_file)

folders = []
# data 폴더 밑에 canny 폴더를 먼저 만들고 수행 
canny_dir = traindata_dir

for i in tqdm(range(len(train_data))):
    class_name = train_data['class_name'].values[i]
    original_image_path = os.path.join(traindata_dir, train_data['image_path'].values[i])
    file_name = train_data['image_path'].values[i].split('/')[-1]
    # print()
    # print(canny_dir)
    # print("file_name",file_name)
    # Canny edge detection 
    image = cv2.imread(original_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 100, 200)

    if class_name in folders:
        cv2.imwrite(canny_dir+'/'+train_data['image_path'].values[i][:-5]+"_edge"+".JPEG", edge)
    else:
        folders.append(class_name)
        #os.makedirs(canny_dir+'/'+class_name)
        cv2.imwrite(canny_dir+'/'+train_data['image_path'].values[i][:-5]+"_edge"+".JPEG", edge)
    
    # if i % 100 == 0:
    #     print(f'{i}th work done')
