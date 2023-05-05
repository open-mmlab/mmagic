# Quick run

After installing MMagic successfully, now you are able to play with MMagic!

To synthesize an image of a church, you only need several lines of codes by MMagic!

```python
from mmagic.apis import init_model, sample_unconditional_model

config_file = 'configs/styleganv2/stylegan2_c2_8xb4-800kiters_lsun-church-256x256.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth'
device = 'cuda:0'
# init a generative model
model = init_model(config_file, checkpoint_file, device=device)
# sample images
fake_imgs = sample_unconditional_model(model, 4)
```

Or you can just run the following command.

```bash
python demo/mmagic_inference_demo_demo.py \
configs/styleganv2/stylegan2_c2_lsun-church_256_b4x8_800k.py \
https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth

```

You will see a new image `unconditional_samples.png` in folder `work_dirs/demos/`, which contained generated samples.

What's more, if you want to make these photos much more clear,
you only need several lines of codes for image super-resolution by MMagic!

```python
import mmcv
from mmagic.apis import init_model, restoration_inference
from mmagic.utils import tensor2img

config = 'configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py'
checkpoint = 'https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth'
img_path = 'tests/data/image/lq/baboon_x4.png'
model = init_model(config, checkpoint)
output = restoration_inference(model, img_path)
output = tensor2img(output)
mmcv.imwrite(output, 'output.png')
```

Now, you can check your fancy photos in `output.png`.
