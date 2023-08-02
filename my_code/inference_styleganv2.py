import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer
import torch
import numpy as np
# Create a MMagicInferencer instance and infer
result_out_dir = './my_code/tmp_res/ema_input_latent.jpg'

# noise和sample_kwargs都是extra_parameters中的参数
# The code snippet you provided is creating some input parameters for the `MMagicInferencer` class.
styles=torch.from_numpy(np.random.RandomState(0).randn(1, 512)).to(torch.float32)
# latent = torch.load('./my_code/latent.pt')[0,0].unsqueeze(0).requires_grad_(True)
# print(latent)
# sample_kwargs = { 'truncation': 1, 'return_noise': True, 'return_features': True, 'input_is_latent': True} # 才是forward函数所输入的参数
sample_kwargs = { 'truncation': 0.7, 'return_noise': True, 'return_features': True, 'input_is_latent': False} # 才是forward函数所输入的参数
extra_parameters={'sample_kwargs': sample_kwargs, 'num_batches': 1, 'noise': styles, 'sample_model': 'ema', 'infer_with_grad': True}
# editor = MMagicInferencer('styleganv2', model_config='', model_ckpt=, model_name=)
editor = MMagicInferencer('styleganv2', 
                          model_setting=6,
                          model_ckpt='./checkpoints/stylegan2_lions_512_pytorch_mmagic.pth',
                          )
# basegan: extra_parameters（最特殊的参数）
results = editor.infer(result_out_dir=result_out_dir, extra_parameters=extra_parameters, infer_with_grad=True)
# results = editor.infer( extra_parameters=extra_parameters)
print(results[0].keys())
img = mmcv.imread(result_out_dir)
plt.imshow(mmcv.bgr2rgb(img))
