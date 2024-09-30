import torch.optim as optim
import torch.nn as nn
from config import config

def get_optimizer(model_parameters):
    if config.OPTIMIZER.lower() == "adam":
        return optim.Adam(model_parameters, lr=config.LEARNING_RATE)
    elif config.OPTIMIZER.lower() == "sgd":
        return optim.SGD(model_parameters, lr=config.LEARNING_RATE)
    # 다른 옵티마이저 추가 가능
    else:
        raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")