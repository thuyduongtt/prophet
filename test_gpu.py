import torch

print('[0] Available GPUs:', torch.cuda.is_available(), torch.cuda.device_count())