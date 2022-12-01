_base_ = '../_base_/gen_default_runtime.py'

model = dict(
    type='EG3D',
    data_preprocessor=dict(type='GenDataPreprocessor'),
    generator=dict(
        type='TriplaneGenerator',
        out_size=512,
        triplane_channels=32,
        triplane_size=256,
        num_mlps=2,
        sr_add_noise=False,
        sr_in_size=128,
        neural_rendering_resolution=128,
        renderer_cfg=dict(
            ray_start=2.25,
            ray_end=3.3,
            box_warp=1,
            depth_resolution=48,
            depth_resolution_importance=48,
            white_back=False,
        ),
        rgb2bgr=True),
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

inception_pkl = './work_dirs/inception_pkl/eg3d_afhq.pkl'
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
    data_root='./data/eg3d/afhq',
    ann_file='afhq.json',
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
