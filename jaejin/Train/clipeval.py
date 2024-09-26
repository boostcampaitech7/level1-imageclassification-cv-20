import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from functions import *
from transformers import CLIPProcessor

def inference(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader
):
    # 모델을 평가 모드로 설정
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            # 데이터를 같은 장치로 이동
            images = images.to(device)
            
            # 모델을 통해 예측 수행
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # 예측 결과 저장
            predictions.extend(preds.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가
    
    return predictions
    
def getClip():
    # 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정
    dir="C:/Users/zin/CV"
    traindata_dir = dir+"/data/train"
    traindata_info_file = dir+"/data/train.csv"

    # 테스트 데이터의 경로와 정보를 가진 파일의 경로를 설정
    testdata_dir = dir+"/data/test"
    testdata_info_file = dir+"/data/test.csv"

    name_info_file = dir+"/data/name_label.csv"
    # 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기
    train_data = pd.read_csv(traindata_info_file)

    # 테스트 데이터
    test_data = pd.read_csv(testdata_info_file)
    name_data = pd.read_csv(name_info_file)

    from transformers import CLIPProcessor, CLIPModel,AdamW
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_result_path = dir+"/clip_1e-05_multiprompt_False_textFrozen_False_Acc_0.8196_best_model.pt"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    state_dict = torch.load(save_result_path)
    model.load_state_dict(state_dict)

    train_info = pd.read_csv(traindata_info_file)

    # 총 class의 수를 측정.
    num_classes = len(train_info['target'].unique())

    transform_selector = TransformSelector(
        transform_type = "AlbumentationsTransformTest"
    )
    train_transform = transform_selector.get_transform(is_train=True)
    test_transform = transform_selector.get_transform(is_train=False)
    
    train_dataset = CLIPDataset(
        root_dir=traindata_dir,
        info_df=train_info,
        transform=train_transform,
        processor=processor,
        device=device,
        use_print = False
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True
    )
    test_info = pd.read_csv(testdata_info_file)
    # 추론에 사용할 Dataset을 선언.
    test_dataset = CLIPDataset(
        root_dir=testdata_dir,
        info_df=test_info,
        transform=test_transform,
        is_inference=True
    )

    # 추론에 사용할 DataLoader를 선언.
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,
        drop_last=False
    )

    scheduler_step_size = 5 # 매 30step마다 학습률 감소
    scheduler_gamma = 0.9  # 학습률을 현재의 90%로 감소

    # 한 epoch당 step 수 계산
    steps_per_epoch = len(train_loader)
    lr = 3e-5
    optimizer = optim.Adam(model.parameters(), lr= lr)

    # 2 epoch마다 학습률을 감소시키는 스케줄러 선언
    epochs_per_lr_decay = 5
    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=scheduler_step_size, 
        gamma=scheduler_gamma
    )
    loss_fn = Loss()
    save_result_path = dir+"/train_result"
    mini_values=get_imagenet_ditction(mini=True,values=True)
    trainer = CLIP_Trainer(
        model=model.to(device), 
        device=device, 
        train_loader=train_loader,
        val_loader=test_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn, 
        processor=processor,
        epochs=100,
        result_path=save_result_path,
        mini_values=mini_values,
        textFrozen=False,
        multi_prompt=True,
        model_name="c",
        lr=lr
    )
# 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
    return trainer#.inference()
# keys_list = list(get_imagenet_ditction(mini=True,values=False).keys())
# tt=[keys_list[index] for index in t]
# test_info['target'] = np.array([name_data[name_data["class_name"] == classes]["target"].values for classes in tt])
# test_info = test_info.reset_index().rename(columns={"index": "ID"})
# #test_info
# #print(test_info['target'])
# # DataFrame 저장
# test_info.to_csv("clipTF7e6_816.csv", index=False)