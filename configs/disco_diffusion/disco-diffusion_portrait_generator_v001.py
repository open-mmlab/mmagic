_base_ = ['./disco-diffusion_adm-u-finetuned_imagenet-512x512.py']
unet_ckpt_path = 'https://download.openmmlab.com/mmediting/synthesizers/disco/adm-u-cvt-rgb_portrait-v001-f4a3f3bc.pth'  # noqa
model = dict(
    unet=dict(base_channels=128),
    secondary_model=None,
    pretrained_cfgs=dict(
        _delete_=True, unet=dict(ckpt_path=unet_ckpt_path, prefix='unet')))
