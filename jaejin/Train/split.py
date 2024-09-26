import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

def splitData(name):
    dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
    traindata_dir = dir+"/data/trainDelFlip_objectsplit"
    traindata_info_file = dir+"/data/"+name
    df = pd.read_csv(traindata_info_file)

    source_folder = traindata_dir
    destination_base = traindata_dir+"_"+name.split(".")[0].split("_")[-1]

    for index, row in df.iterrows():
        filename = row['class_name']  # CSV에서 파일 이름을 포함하는 열의 이름
        category = row['image_path'].split('/')[-1]  # CSV에서 분류 기준이 되는 열의 이름
  
        source_path = os.path.join(source_folder, filename, category)
        destination_folder = os.path.join(destination_base, filename)#.split('/')[:-2]
        destination_path = os.path.join(destination_folder, category)
 
        # 대상 폴더가 없으면 생성
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)        

        # 파일 복사 또는 이동
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)  # 복사하려면 copy2 사용
            # shutil.move(source_path, destination_path)  # 이동하려면 move 사용
        else:
            print(f"File not found: {source_path}")

splitData("trainDelFlip_objectsplit_val.csv")
splitData("trainDelFlip_objectsplit_train.csv")
def makeSplitFile():
    dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
    traindata_info_file = dir+"/data/trainDelFlip_objectsplit.csv"

    train_info = pd.read_csv(traindata_info_file)
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        random_state=20,
        stratify=train_info['target']
        )
    
    pd.DataFrame(train_df).to_csv('trainDelFlip_objectsplit_train.csv', index=False)
    pd.DataFrame(val_df).to_csv('trainDelFlip_objectsplit_val.csv', index=False)


# 나눈거 저장
# makeSplitFile()


# 이건 클래스별 수 볼 떄
# dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
# data_dir = dir+"/data/trainDelFlip_objectsplit"
# traindata_info_file = dir+"/data/trainDelFlip_objectsplit.csv"

# train_info = pd.read_csv(traindata_info_file)

# #print(train_info["target"].unique())

# class_names = os.listdir(data_dir)

# class_distribution = {class_name: len(os.listdir(os.path.join(data_dir, class_name))) for class_name in class_names if class_name[0] == 'n'}
# import numpy as np
# # 목표 이미지 수 설정 (예: 가장 많은 클래스의 이미지 수)
# target_count = sorted(class_distribution.values())
# print(target_count)

# for class_name in class_names:
#     if len(os.listdir(os.path.join(data_dir, class_name))) == 1 and class_name[0] == 'n':
#         print(class_name)