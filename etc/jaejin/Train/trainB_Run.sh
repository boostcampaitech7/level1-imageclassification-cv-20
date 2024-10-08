: << "END"
variable1="resnet18"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"
variable1="resnet50"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"
variable1="resnet101"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"

variable1="densenet121"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"
variable1="inception_v3"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"
variable1="mobilenet_v2"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"
variable1="efficientnet_b1"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"

variable1="vit_b_16"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"
variable1="vit_l_16"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"

variable1="convnext_base"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"
variable1="convnext_large"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"
variable1="convnext_tiny"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"

variable1="dino-vitb8"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"
END


variable1="dino-vitb8"
python train_baseline_ag.py "$variable1" "$variable2" "$variable3"
END
