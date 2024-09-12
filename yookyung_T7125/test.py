import os

print(len(os.listdir('./data/test')))
total = 0
print(len(os.listdir('./data/train')))

array = []
for folder in os.listdir('./data/train'):
    path = os.path.join('./data/train',folder)
    array.append(len(os.listdir(path)))
    for file in os.listdir(path):
        if not file.endswith('JPEG'):
            print(folder)


print(total)
print(sum(array))