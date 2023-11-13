import torch

print('Available GPUs:', torch.cuda.is_available(), torch.cuda.device_count())