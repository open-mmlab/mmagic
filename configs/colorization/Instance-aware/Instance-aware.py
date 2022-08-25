model = dict(
    type='FusionModel',
    ab_norm=110.,
    l_norm=100.,
    l_cent=50.,
    sample_Ps=[
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
    mask_cent=.5,
    init_type='normal',
    fusion_weight_path='../checkpoints/coco_finetuned_mask_256_ffs',
    which_direction='AtoB',
    instance_model=dict(
        type='InstanceGenerator',
        input_nc=4,
        output_nc=2,
    ),
    full_model=dict(
        type='SIGGRAPHGenerator',
        input_nc=4,
        output_nc=2,
    ),
    fusion_model=dict(
        type='FusionGenerator',
        input_nc=4,
        output_nc=2,
    ))

test_cfg = dict(metrics=['psnr', 'ssim'])

test_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='GenMaskRCNNBbox', stage='test_fusion', finesize=256),
    dict(type='Resize', keys=['gt', 'cropped_img']),
    dict(
        type='ImageToTensor',
        keys=[
            'cropped_img', 'box_info', 'box_info_2x', 'box_info_4x',
            'box_info_8x'
        ])
]
