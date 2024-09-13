import timm
print(timm.list_models(pretrained=True))

import torch
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())