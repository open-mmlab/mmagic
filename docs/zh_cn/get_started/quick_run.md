# 快速运行

成功安装MMagic后，现在您可以玩转MMagic了！如果您要从文本生成图像，只需要MMagic的几行代码！

```python
from mmagic.apis import MMagicInferencer
sd_inferencer = MMagicInferencer(model_name='stable_diffusion')
text_prompts = 'A panda is having dinner at KFC'
result_out_dir = 'output/sd_res.png'
sd_inferencer.infer(text=text_prompts, result_out_dir=result_out_dir)
```

或者您可以运行以下命令。

```bash
python demo/mmagic_inference_demo.py \
    --model-name stable_diffusion \
    --text "A panda is having dinner at KFC" \
    --result-out-dir ./output/sd_res.png
```

您将在文件夹`output/`中看到一个新图像`sd_res.png`，其中包含生成的样本。

更重要的是，如果您想让这些照片更清晰，MMagic的超分辨率只需要几行代码！

```python
from mmagic.apis import MMagicInferencer
config = 'configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py'
checkpoint = 'https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth'
img_path = 'tests/data/image/lq/baboon_x4.png'
editor = MMagicInferencer('esrgan', model_config=config, model_ckpt=checkpoint)
output = editor.infer(img=img_path,result_out_dir='output.png')
```

现在，您可以在 `output.png` 中查看您想要的图片。
