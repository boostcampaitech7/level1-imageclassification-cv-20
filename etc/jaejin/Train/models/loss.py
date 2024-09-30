import torch
import torch.nn as nn
from config import config
import torch.functional as F

def focal_loss(
    outputs: torch.Tensor, 
    targets: torch.Tensor, 
    alpha: float = 0.25, 
    gamma: float = 2.0, 
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Focal Loss 계산 함수.
    """
    ce_loss = F.cross_entropy(outputs, targets, reduction='none')  # 기본 Cross Entropy
    pt = torch.exp(-ce_loss)  # P_t, 예측이 맞은 확률
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss  # Focal Loss 계산

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss
    
def get_loss_function():
    """
    Config에서 설정을 읽어 손실함수를 생성하고 반환하는 함수.
    """
    if config.LOSS == "cross_entropy":
        return nn.CrossEntropyLoss(**config.LOSS_PARAMS["cross_entropy"])
    elif config.LOSS == 'focal_loss':
        return focal_loss()
    elif config.LOSS == "bce_with_logits":
        return nn.BCEWithLogitsLoss(**config.LOSS_PARAMS["bce_with_logits"])
    elif config.LOSS == "mse":
        return nn.MSELoss(**config.LOSS_PARAMS["mse"])
    elif config.LOSS == 'kldivloss':
        return nn.KLDivLoss()
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