#

variable1="resnet18"
python baseline_pretrained_trained_evaluate.py "$variable1"
: << "END"
variable1="resnet50"
python baseline_pretrained_trained_evaluate.py "$variable1"
variable1="resnet101"
python baseline_pretrained_trained_evaluate.py "$variable1"

variable1="densenet121"
python baseline_pretrained_trained_evaluate.py "$variable1"
variable1="inception_v3"
python baseline_pretrained_trained_evaluate.py "$variable1"
variable1="mobilenet_v2"
python baseline_pretrained_trained_evaluate.py "$variable1"
variable1="efficientnet_b0"
python baseline_pretrained_trained_evaluate.py "$variable1"

variable1="vit_b_16"
python baseline_pretrained_trained_evaluate.py "$variable1"
variable1="vit_l_16"
python baseline_pretrained_trained_evaluate.py "$variable1"
END



