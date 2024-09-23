import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import pandas as pd
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(self.data_info['class_name'].unique())}

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_info.iloc[idx, 1])
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[self.data_info.iloc[idx, 0]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CutMixDataset(Dataset):
    def __init__(self, dataset, num_classes, num_mix, alpha=1.0):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_mix = num_mix
        self.cutmix = v2.CutMix(num_classes=num_classes, alpha=alpha)

    def __len__(self):
        return len(self.dataset) * self.num_mix

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx % len(self.dataset)]
        img2, label2 = self.dataset[torch.randint(len(self.dataset), (1,)).item()]

        img1 = img1.unsqueeze(0)  # Add batch dimension
        img2 = img2.unsqueeze(0)
        label1 = torch.tensor([label1])
        label2 = torch.tensor([label2])

        cutmix_imgs, cutmix_labels = self.cutmix(torch.cat([img1, img2]), torch.cat([label1, label2]))
        
        return cutmix_imgs[0], cutmix_labels[0]  # Remove batch dimension

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 원본 데이터셋 생성
csv_file = '/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/yookyung/data/train.csv'
root_dir = '/data/ephemeral/home/cv20-proj1/level1-imageclassification-cv-20/yookyung/data/train/'
original_dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

# CutMix 데이터셋 생성
num_classes = len(original_dataset.class_to_idx)
cutmix_dataset = CutMixDataset(original_dataset, num_classes=num_classes, num_mix=1)

# 원본 데이터셋과 CutMix 데이터셋 결합
combined_dataset = ConcatDataset([original_dataset, cutmix_dataset])

# 데이터 로더 생성
train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=4)

# # 모델 정의 (예시)
# import torch.nn as nn
# import torch.optim as optim

# class SimpleModel(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleModel, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.classifier = nn.Linear(128 * 56 * 56, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

# # 모델, 손실 함수, 옵티마이저 설정
# model = SimpleModel(num_classes)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 학습 루프
# num_epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# for epoch in range(num_epochs):
#     model.train()
#     for batch_idx, (images, labels) in enumerate(train_loader):
#         images, labels = images.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         if batch_idx % 100 == 0:
#             print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

# print("Training finished!")