_base_ = '../_base_/gen_default_runtime.py'

model = dict(
    type='EG3D',
    data_preprocessor=dict(type='GenDataPreprocessor'),
    generator=dict(
        type='TriplaneGenerator',
        out_size=128,
        zero_cond_input=True,
        cond_scale=0,
        sr_in_size=64,
        renderer_cfg=dict(
            # Official implementation set ray_start, ray_end and box_warp as
            # 0.1, 2.6 and 1.6 respectively, and FID is 7.2441
            # ray_start=0.1,
            # ray_end=2.6,
            # box_warp=1.6,
            ray_start=0.4,
            ray_end=2.0,
            box_warp=1.7,
            depth_resolution=64,
            depth_resolution_importance=64,
            white_back=True,
        ),
        rgb2bgr=True),
    camera=dict(
        type='UniformCamera',
        horizontal_mean=3.141,
        horizontal_std=3.141,
        vertical_mean=3.141 / 2,
        vertical_std=3.141 / 2,
        focal=1.025390625,
        up=[0, 0, 1],
        radius=1.2),
)

train_cfg = train_dataloader = optim_wrapper = None
val_cfg = val_dataloader = val_evaluator = None

inception_pkl = './work_dirs/inception_pkl/eg3d_shapenet.pkl'
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
        sample_model='orig'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color'),
    dict(type='PackEditInputs')
]
test_dataset = dict(
    type='BasicConditionalDataset',
    data_root='./data/eg3d/shapenet-car',
    ann_file='shapenet.json',
    pipeline=test_pipeline)
test_dataloader = dict(
    # NOTE: `batch_size = 16` cost nearly **12GB** of GPU memory,
    # modification this param by yourself corresponding to your own GPU.
    batch_size=16,
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
        # save_at_test=False,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]
