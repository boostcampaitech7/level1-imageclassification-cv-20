import torch
import torch.nn as nn
from config import config

def get_loss_function():
    """
    Config에서 설정을 읽어 손실함수를 생성하고 반환하는 함수.
    """
    if config.LOSS == "cross_entropy":
        return nn.CrossEntropyLoss(**config.LOSS_PARAMS["cross_entropy"])
    elif config.LOSS == "bce_with_logits":
        return nn.BCEWithLogitsLoss(**config.LOSS_PARAMS["bce_with_logits"])
    elif config.LOSS == "mse":
        return nn.MSELoss(**config.LOSS_PARAMS["mse"])
    # 필요한 다른 손실 함수들 추가
    else:
        raise ValueError(f"Unsupported loss function: {config.LOSS}")
    
class Loss(nn.Module):
    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = get_loss_function()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        return self.loss_fn(outputs, targets)