import pandas as pd
import os
def getTargetInfo():
    dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20"
    traindata_info_file = dir+"/data/train.csv"
    train_info = pd.read_csv(traindata_info_file)
    classTarget ={}
    for class_name in sorted(train_info["class_name"].unique()):
        classTarget[class_name] = train_info[train_info["class_name"] == class_name]["target"].unique()[0]
    return classTarget

def makeInfo(name):
    dir="/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/data/"
    balanced_dataset = dir+name
    balanced_dataset_info = []
    classTarget = getTargetInfo()
    for class_name in os.listdir(balanced_dataset):
        class_dir = os.path.join(balanced_dataset, class_name)
        classN = class_dir.split('/')[-1]
        
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if os.path.isfile(file_path):
                    balanced_dataset_info.append({
                        'class_name': classN,
                        'image_path': classN+"/"+file_path.split('/')[-1],
                        'target': classTarget[classN]
                    })
    pd.DataFrame(balanced_dataset_info).to_csv(name+'.csv', index=False)

makeInfo("trainDelFlip_objectsplit_train_up_canny")
makeInfo("trainDelFlip_objectsplit_train_upWithcanny")
