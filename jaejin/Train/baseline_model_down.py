import sys
from torchvision import models

name = sys.argv[1]
version = sys.argv[2]

if name == "Resnet":
    if version == "18":
        M = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif version == "50":
        M = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif version == "101":
        M = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    else:
        print("wrong version!!!")
        sys.exit(0)
elif name == "DenseNet":
    M = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

elif name == "Inception":
    M = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

elif name == "MobileNet":
    M = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

elif name == "EfficientNet":
    M = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)

elif name == "ViT":
    if version == "B/16":
        M = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    elif version == "L/16":
        M = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)    
    else:
        print("wrong version!!!")
        sys.exit(0)
elif name == "convnext_base":
    M = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)

elif name == "convnext_large":
    M = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)

elif name == "convnext_tiny":
    M = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
else:
    print("wrong version!!!")
    sys.exit(0)

print()
if version == "n":
    print("complete model download",name)
else:
    print("complete model download",name,version)