import torch

ckpt = torch.load("work_dirs/vico/iter_400.pth", map_location="cpu")

for k, v in ckpt['state_dict'].items():
    print(k)