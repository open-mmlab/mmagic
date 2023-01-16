model = dict(
    type='GFPGAN',
    generator=dict(
        type='GFPGANv1Clean',
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        sft_half=True),
    face_restore_cfg=dict(upscale=2, device='cuda'))
