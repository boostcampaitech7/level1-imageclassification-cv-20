# import os

# print(len(os.listdir('./data/test')))
# total = 0
# print(len(os.listdir('./data/train')))

# array = []
# for folder in os.listdir('./data/train'):
#     path = os.path.join('./data/train',folder)
#     array.append(len(os.listdir(path)))
#     for file in os.listdir(path):
#         if not file.endswith('JPEG'):
#             print(folder)


# print(total)
# print(sum(array))
import pandas as pd
import os

dir='/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20'
# 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정.
traindata_dir = dir+"/data/train"
traindata_info_file = dir+"/data/train.csv"
save_result_path = dir+"/train_result"
print(os.getcwd())
print(traindata_info_file)

# traindata_info_file = '/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/data/train.csv'
# 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
train_info = pd.read_csv(traindata_info_file)

print(train_info.head())

