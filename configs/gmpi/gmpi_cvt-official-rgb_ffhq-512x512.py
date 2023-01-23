_base_ = '../_base_/gen_default_runtime.py'

model = dict(
    type='EG3D',
    data_preprocessor=dict(type='GenDataPreprocessor'),
    generator=dict(
        type='GMPIGenerator',
        latent_dim=512,
        generator_label_dim=0,
        stylegan2_w_dim=512,
        img_resolution=512,
        mapping_kwargs=dict(num_layers=8),
        synthesis_kwargs=dict(
            channel_base=32768,
            channel_max=512,
            conv_clamp=256,
            num_fp16_res=4),
        pos_enc_multires=0,
        background_alpha_full=True,
        torgba_sep_background=True,
        build_background_from_rgb=True,
        build_background_from_rgb_ratio=0.05,
        cond_on_pos_enc_only_alpha=True,
        gen_alpha_largest_res=256,
        G_final_img_act='tanh',
    ),
    camera=dict(
        type='GaussianCamera',
        horizontal_mean=3.14 / 2,
        horizontal_std=0.35,
        vertical_mean=3.14 / 2 - 0.05,
        vertical_std=0.25,
        radius=2.7,
        fov=18.837,
        look_at=[0, 0, 0.2]))

train_cfg = train_dataloader = optim_wrapper = None
val_cfg = val_dataloader = val_evaluator = None

inception_pkl = './work_dirs/inception_pkl/eg3d_ffhq_512.pkl'
metrics = [
    dict(
        type='FID-Full',
        prefix='FID-Full',
        fake_nums=50000,
        inception_pkl=inception_pkl,
        need_cond_input=True,
        sample_model='orig'),
    dict(
        type='FID-Full',
        prefix='FID-Random-Camera',
        fake_nums=50000,
        inception_pkl=inception_pkl,
        sample_model='orig')
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color'),
    dict(type='PackEditInputs')
]
test_dataset = dict(
    type='BasicConditionalDataset',
    data_root='./data/eg3d/ffhq_512',
    ann_file='ffhq_512.json',
    pipeline=test_pipeline)
test_dataloader = dict(
    # NOTE: `batch_size = 4` cost nearly **9.5GB** of GPU memory,
    # modification this param by yourself corresponding to your own GPU.
    batch_size=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    num_workers=9,
    dataset=test_dataset)

test_evaluator = dict(metrics=metrics)

custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]
