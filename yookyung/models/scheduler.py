from torch.optim import lr_scheduler
from config import config

def get_scheduler(scheduler_name, optimizer, steps_per_epoch):
    """
    Config에서 설정을 읽어 스케쥴러 생성하고 반환하는 함수.
    """
    if scheduler_name == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=config.SCHEDULER_PARAMS["step"]["step_size"],
            gamma=config.SCHEDULER_PARAMS["step"]["gamma"]
        )
    elif scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.SCHEDULER_PARAMS["cosine"]["T_max"],
            eta_min=config.SCHEDULER_PARAMS["cosine"]["eta_min"]
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
