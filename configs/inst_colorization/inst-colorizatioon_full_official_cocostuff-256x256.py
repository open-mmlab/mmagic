_base_ = ['../_base_/default_runtime.py']

experiment_name = 'inst-colorization_full_official_cocostuff_256x256'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

stage = 'full'

model = dict(
    type='InstColorization',
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[127.5],
        std=[127.5],
    ),
    image_model=dict(
        type='ColorizationNet', input_nc=4, output_nc=2, norm_type='batch'),
    instance_model=dict(
        type='ColorizationNet', input_nc=4, output_nc=2, norm_type='batch'),
    fusion_model=dict(
        type='FusionNet', input_nc=4, output_nc=2, norm_type='batch'),
    color_data_opt=dict(
        ab_thresh=0,
        p=1.0,
        sample_PS=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ],
        ab_norm=110,
        ab_max=110.,
        ab_quant=10.,
        l_norm=100.,
        l_cent=50.,
        mask_cent=0.5),
    which_direction='AtoB',
    loss=dict(type='HuberLoss', delta=.01))

# yapf: disable
test_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(
        type='InstanceCrop',
        config_file='mmdet::mask_rcnn/mask-rcnn_x101-32x8d_fpn_ms-poly-3x_coco.py',  # noqa
        finesize=256,
        box_num_upbound=5),
    dict(
        type='Resize',
        keys=['img', 'cropped_img'],
        scale=(256, 256),
        keep_ratio=False),
    dict(type='PackEditInputs'),
]
