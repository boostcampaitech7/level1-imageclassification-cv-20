import os
import pandas as pd

dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
traindata_dir = dir+"/data/train"
traindata_info_file = dir+"/data/train.csv"

testdata_dir = dir+"/data/test"
testdata_info_file = dir+"/data/test.csv"

train_data = pd.read_csv(traindata_info_file)
test_data = pd.read_csv(testdata_info_file)

path = test_data['image_path'].tolist()
temp = -1
start = False
for index in range(0,10014):
    if not os.path.exists(testdata_dir+"/"+path[index]):
        if not start:
            index
            start = True
            print(index)
            if temp != -1:
                print()
            temp = index
    else:
        if start:
            start = False
            if temp != index-1:
                print(index-1)
            