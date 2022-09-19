model = dict(
    type='FusionModelTest',
    ngf=64,
    output_nc=2,
    # avg_loss_alpha=.986,
    ab_norm=110.,
    l_norm=100.,
    l_cent=50.,
    sample_Ps=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    mask_cent=.5,
    init_type='normal',
    fusion_weight_path='/mnt/ruoning/coco_finetuned_mask_256_ffs',
    resize_or_crop='resize_or_crop',
    which_direction='AtoB',
    instance_model=dict(
        type='InstanceGenerator',
        input_nc=4,
        output_nc=2,
        norm_type='batch',
        classification=False),
    fusion_model=dict(
        type='FusionGenerator',
        input_nc=4,
        output_nc=2,
        norm_type='batch',
        classification=False))

test_cfg = dict(metrics=['psnr', 'ssim'])
test_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='GenMaskRCNNBbox', stage='test_fusion', finesize=256),
    dict(type='Resize', keys=['gt', 'cropped_img']),
    dict(
        type='Collect',
        keys=[
            'full_img',
            'empty_box',
            'cropped_img',
            'box_info',
            'box_info_2x',
            'box_info_4x',
            'box_info_8x',
        ]),
    dict(
        type='ImageToTensor',
        keys=[
            'full_img', 'cropped_img', 'box_info', 'box_info_2x',
            'box_info_4x', 'box_info_8x'
        ]),
]
