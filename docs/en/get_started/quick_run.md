# Quick run

After installing MMagic successfully, now you are able to play with MMagic! To generate an image from text, you only need several lines of codes by MMagic!

```python
from mmagic.apis import MMagicInferencer
sd_inferencer = MMagicInferencer(model_name='stable_diffusion')
text_prompts = 'A panda is having dinner at KFC'
result_out_dir = 'output/sd_res.png'
sd_inferencer.infer(text=text_prompts, result_out_dir=result_out_dir)
```

Or you can just run the following command.

```bash
python demo/mmagic_inference_demo.py \
    --model-name stable_diffusion \
    --text "A panda is having dinner at KFC" \
    --result-out-dir ./output/sd_res.png
```

You will see a new image `sd_res.png` in folder `output/`, which contained generated samples.

What's more, if you want to make these photos much more clear,
you only need several lines of codes for image super-resolution by MMagic!

```python
from mmagic.apis import MMagicInferencer
config = 'configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py'
checkpoint = 'https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth'
img_path = 'tests/data/image/lq/baboon_x4.png'
editor = MMagicInferencer('esrgan', model_config=config, model_ckpt=checkpoint)
output = editor.infer(img=img_path,result_out_dir='output.png')
```

Now, you can check your fancy photos in `output.png`.
