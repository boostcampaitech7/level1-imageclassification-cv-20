import numpy as np
import pandas as pd
import os

dir = '/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/data/trainDelFlip_objectsplit_val'
csv = pd.read_csv('/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/data/trainDelFlip_objectsplit_val.csv')

group_count = 0
content = 0
for folder in os.listdir(dir):
    group_count += 1
    for file in os.listdir(os.path.join(dir, folder)):
        if not file.endswith('JPEG'):
            print(folder, file)
            content += 1

print(len(csv))
print(content)
print(group_count)