import torch
from tqdm import tqdm

ckpt = torch.load('checkpoint/NAFNet-SIDD-width64.pth')
print(ckpt.keys())
params = ckpt['params']
keys = params.keys()
print(list(keys)[:20])
for key in tqdm(list(keys)):
    new_key = 'generator.'+key
    # new_ema_key = 'generator_ema.'+key
    params[new_key] = params.pop(key)
    # params[new_ema_key] = params[new_key]

path = 'checkpoint/NAFNet-SIDD-midc64.pth'
torch.save(params, path)
print(list(params.keys())[:20])