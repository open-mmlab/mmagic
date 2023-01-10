_base_ = ['configs/guided_diffusion/adm-u_ddim250_8xb32_imagenet-64x64.py']

 
model = dict(
    classifier = dict(type = 'EncoderUNetModel',
        image_size=64,
        in_channels=3,
        model_channels=128,
        out_channels=1000,
        num_res_blocks=4,
        attention_resolutions=(2,4,8),
        channel_mult=(1,2,3,4),
        use_fp16=False,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
        pool='attention')   
)