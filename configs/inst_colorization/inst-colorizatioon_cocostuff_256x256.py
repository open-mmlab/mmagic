from logging import PlaceHolder

_base_ = ['../_base_/default_runtime.py']

exp_name = 'Instance-aware_full'
save_dir = './'
work_dir = '..'

stage = 'test'
model = dict(
    type='InstColorization',
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[127.5],
        std=[127.5],
    ),
    generator=dict(
        type='InstColorizationGenerator',
        stage=stage,
        detector=PlaceHolder,
        full_model=dict(
            type='InstanceGenerator',
            input_nc=4,
            output_nc=2,
            norm_type='batch'),
        instance_model=dict(
            type='InstanceGenerator',
            input_nc=4,
            output_nc=2,
            norm_type='batch'),
        fusion_model=dict(
            type='FusionGenerator', input_nc=4, output_nc=2,
            norm_type='batch')),
    insta_stage=stage,
    ngf=64,
    output_nc=2,
    avg_loss_alpha=.986,
    ab_norm=110.,
    ab_max=110.,
    ab_quant=10.,
    l_norm=100.,
    l_cent=50.,
    sample_Ps=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    mask_cent=.5,
    which_direction='AtoB',
    loss=dict(type='HuberLoss', delta=.01))

test_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='GenMaskRCNNBbox', stage=stage, finesize=256),
    dict(type='Resize', keys=['img'], scale=(256, 256), keep_ratio=False),
    dict(type='PackEditInputs'),
]
