# Sketch 이미지 데이터 분류

## 1. 프로젝트 소개

<p align="center"><img src="https://github.com/user-attachments/assets/be91a81c-d699-4278-86c2-382a1edf6037" width='800' height='300'></p>

컴퓨터 비전 분야에서는 다양한 형태의 이미지 데이터에 대해 인식과 분류와 같은 문제를 해결하고 있습니다. 여러 형태의 이미지 중 스케치 데이터는 사진과 다르게 사물에 대한 개념적 이해를 바탕으로 색상, 질감 등 세부적인 특성이 단순화된 형태의 데이터입니다. 색상, 질감 등 세부적인 특성들이 결여되어 있고, 대상의 형태와 구조에 초점을 맞춘 경우가 많습니다.<br/>

본 프로젝트에서는 이러한 스케치 데이터의 특성을 이해하고, 다양한 모델이 객체의 특징을 학습하여 적합한 클래스로 분류하는 것을 목적으로 하고 있습니다. 이를 통해 모델이 객체의 기본적인 형태와 구조를 학습하고 인식하도록 함으로써, 비전 시스템의 범용성을 향상시키고 다양한 실제 어플리케이션에 적용할 수 있는 능력을 키우고자 합니다. <br/>

프로젝트 기간 : 24.09.10 ~ 24.09.26

```
부스트코스 강의 수강 및 과제 : 24.09.10 ~ 24.09.15
데이터 EDA 확인 및 베이스라인 모델 학습 : 24.09.16 ~ 24.09.18
데이터 전처리 및 파인튜닝 : 24.09.19 ~ 24.09.24
앙상블 및 제출 : 24.09.25 ~ 24.09.26
```
<br/>

## 2.🧑‍🤝‍🧑 Team ( CV-20 : CV Up!!)

<div align=center>

|<img src="https://avatars.githubusercontent.com/PollinKim" width='80'>|<img src="https://avatars.githubusercontent.com/kaeh3403" width="80"> |<img src="https://avatars.githubusercontent.com/sweetpotato15" width="80">|<img src="https://avatars.githubusercontent.com/jeajin" width='80'>|<img src="https://avatars.githubusercontent.com/SuyoungPark11" width='80'>|<img src="https://avatars.githubusercontent.com/uddaniiii" width='80'>|
|:---:|:---:|:---:|:---:|:---:|:---:|
[**김경수**](https://github.com/PollinKim) |[**김성규**](https://github.com/kaeh3403) | [**김유경**](https://github.com/sweetpotato15) | [**김재진**](https://github.com/jeajin) | [**박수영**](https://github.com/SuyoungPark11) | [**이단유**](https://github.com/uddaniiii)
T7174|T7118|T7125|T7127|T7150|T7217|
|데이터 전처리<br>협업 툴 관리|데이터 전처리<br>모델 적용|데이터 증강<br>모델 적용| 개발 환경 세팅<br>모델 적용|데이터 증강<br>모델 적용|데이터셋 분석<br>모델 적용

</div>

wrap up 레포트 : [wrap up report](https://)
<br/>
<br/>

## 3. EDA

본 프로젝트에서 사용된 데이터셋은 ImageNet Sketch 데이터셋 중 500개 클래스에 해당하는 총 25,035개의 데이터로 구성되어 있습니다. 

### 3-1. 데이터셋 재구성
다만, 데이터셋 전수 조사 한 결과 

<p align="center"><img src="https://github.com/user-attachments/assets/9f9e06c0-cb1f-414a-a3d5-9b561f38bc86"></p>

> - 동일 클래스에 동일한 이미지가 flip, resize 된 사례 관찰 (위 사진 속 좌측 이미지)
>
> - 이미지 하나에 여러 객체가 포함된 사례 관찰 (위 사진 속 우측 이미지)

을 확인 할 수 있었고, 해당 클래스의 다양한 형태를 학습하지 못하거나 잘못된 특징을 학습할 우려가 있다고 판단하여 전처리 진행하였습니다. 

> - delFlip : flip된 동일한 이미지를 제거한 데이터셋 
>
> - objectsplit : 여러 객체가 포함된 이미지를 분리한 데이터셋
>
> - delFlip_objectsplit : 위의 두 특성을 합쳐 RandAug 기법으로 upsampling 한 데이터셋
>
> - delFlip_objectsplit_upwithcanny : 위의 데이터셋을 canny 엣지 검출 적용하여 증강한 데이터셋

으로 구성하여 다양하게 실험을 진행했습니다. 

### 3-2. 증강 기법 적용

또한, 이미지의 다양한 특징을 학습하고 모델의 정확도와 강건성을 향상시키기 위해 아래와 같은 증강 기법을 적용하였습니다. 

- canny 엣지 검출 

- cutmix 적용

<p align="center"><img src="https://github.com/user-attachments/assets/ed05ad2f-405a-426c-8b06-797b7fdd4240" width='400' height='150'></p> 

<br/>
<br/>

## 4. 실험 

### 4-1. 모델별 성능 비교

| model name | learning rate | accuracy |
|:---:|:---:|:---:|
|**efficient_b1** |0.0003|**0.7823**|
|resnet18|0.0003|0.4329|
|resnet50|0.0003|0.6279|
|resnet101|0.0003|0.5956|
|vit_b_16|0.0003|0.6575|
|densenet121|0.0003|0.4552|
|mobile_v2|0.0003|0.576|
<br/>

### 4-2. 증강 기법별 ConvNext 성능 비교

|sketch_ab|adge|cutmix|NewDataset| learning rate | accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|
|-|-|-|-|0.0003|0.8224|
|-|v|-|-|0.0003|0.8690|
|v|-|-|-|0.0003|0.8840|
|v|v|-|-|0.0003|0.8770|
|**v**|-|**v**|-|0.0003|**0.8880**|
|v|-|v|v|0.0003 (3epoch)|0.8850|

`NewDataset` : delFlip_objectsplit_upwithcanny

`sketch_ab` : Affine + Elastic deformation + Gaussian noise + Motion Blur

<br/>

### 4-3. CLIP 모델별 성능 비교

|text freezing |multi prompt| learning rate | accuracy |
|:---:|:---:|:---:|:---:|
|v|-|0.000005|0.7973|
|v|-|0.000003|0.8003|
|**-**|**-**|0.00001|**0.8190**|
|-|-|0.000005|0.8170|
|-|v|0.00003|0.7604|
|-|v|0.000003|0.7717|
|-|v|0.000007|0.7810|
<br/>

### 4-4. DINO 모델별 성능 비교

|classifier|aug|learning rate|accuracy|
|:-:|:-:|:-:|:-:|
|**linear**|**-**|**0.0003**|**0.7477**|
||-|0.0008|0.7462|
||-|0.00003|0.5557|
|mlp|-|0.00003|0.6013|
|attention|-|0.00003|0.7225|
||v|0.003|0.6825|

`aug = False` : 10 epoch 까지만 수행
<br/>
<br/>

### 4-5. 앙상블 성능 비교

|method|model1|model2|model3|model4|accuracy|
|:-:|:-:|:-:|:-:|:-:|:-:|
|**soft voting**|**convnext (edge+sketch ab)**|**convnext (cutmix+sketch ab)**|||**0.8940**|
|genetic|convnext (sketch ab)|clip (basic aug)|dino|efficient|0.8420|
|bayesian|convnext (sketch ab)|clip (basic aug)|dino|efficient|0.8820|
|random forest|convnext (sketch ab)|clip (basic aug)|dino|efficient|0.8442|
<br/>

## 5. 개발 환경
- 버전 관리 : Github

- 협업 툴 : Notion, Spread sheet, Slack, Wandb

- 서버 : V100 GPU
<br/>
<br/>

## 6. 프로젝트 구조

```
📦level1-imageclassification-cv-20
├-ㅡ
│
│
└


```
