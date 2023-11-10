# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler

from mmagic.datasets import BasicFramesDataset
from mmagic.datasets.transforms import (GenerateSegmentIndices,
                                        LoadImageFromFile, MirrorSequence,
                                        PackInputs)
from mmagic.engine.runner import MultiTestLoop
from mmagic.evaluation import PSNR, SSIM

# configs for REDS4
reds_data_root = 'data/REDS'

reds_pipeline = [
    dict(type=GenerateSegmentIndices, interval_list=[1]),
    dict(type=LoadImageFromFile, key='img', channel_order='rgb'),
    dict(type=LoadImageFromFile, key='gt', channel_order='rgb'),
    dict(type=PackInputs)
]

reds_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicFramesDataset,
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root=reds_data_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_val.txt',
        depth=1,
        num_input_frames=100,
        fixed_seq_len=100,
        pipeline=reds_pipeline))

reds_evaluator = [
    dict(type=PSNR, prefix='REDS4-BIx4-RGB'),
    dict(type=SSIM, prefix='REDS4-BIx4-RGB')
]

# configs for vimeo90k-bd and vimeo90k-bi
vimeo_90k_data_root = 'data/vimeo90k'
vimeo_90k_file_list = [
    'im1.png', 'im2.png', 'im3.png', 'im4.png', 'im5.png', 'im6.png', 'im7.png'
]

vimeo_90k_pipeline = [
    dict(type=LoadImageFromFile, key='img', channel_order='rgb'),
    dict(type=LoadImageFromFile, key='gt', channel_order='rgb'),
    dict(type=MirrorSequence, keys=['img']),
    dict(type=PackInputs)
]

vimeo_90k_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicFramesDataset,
        metainfo=dict(dataset_type='vimeo90k_seq', task_name='vsr'),
        data_root=vimeo_90k_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vimeo90K_test_GT.txt',
        depth=2,
        num_input_frames=7,
        fixed_seq_len=7,
        load_frames_list=dict(img=vimeo_90k_file_list, gt=['im4.png']),
        pipeline=vimeo_90k_pipeline))

vimeo_90k_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicFramesDataset,
        metainfo=dict(dataset_type='vimeo90k_seq', task_name='vsr'),
        data_root=vimeo_90k_data_root,
        data_prefix=dict(img='BIx4', gt='GT'),
        ann_file='meta_info_Vimeo90K_test_GT.txt',
        depth=2,
        num_input_frames=7,
        fixed_seq_len=7,
        load_frames_list=dict(img=vimeo_90k_file_list, gt=['im4.png']),
        pipeline=vimeo_90k_pipeline))

vimeo_90k_bd_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='Vimeo-90K-T-BDx4-Y'),
    dict(type=SSIM, convert_to='Y', prefix='Vimeo-90K-T-BDx4-Y'),
]

vimeo_90k_bi_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='Vimeo-90K-T-BIx4-Y'),
    dict(type=SSIM, convert_to='Y', prefix='Vimeo-90K-T-BIx4-Y'),
]

# config for UDM10 (BDx4)
udm10_data_root = 'data/UDM10'

udm10_pipeline = [
    dict(
        type=GenerateSegmentIndices,
        interval_list=[1],
        filename_tmpl='{:04d}.png'),
    dict(type=LoadImageFromFile, key='img', channel_order='rgb'),
    dict(type=LoadImageFromFile, key='gt', channel_order='rgb'),
    dict(type=PackInputs)
]

udm10_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicFramesDataset,
        metainfo=dict(dataset_type='udm10', task_name='vsr'),
        data_root=udm10_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        pipeline=udm10_pipeline))

udm10_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='UDM10-BDx4-Y'),
    dict(type=SSIM, convert_to='Y', prefix='UDM10-BDx4-Y')
]

# config for vid4
vid4_data_root = 'data/Vid4'

vid4_pipeline = [
    dict(type=GenerateSegmentIndices, interval_list=[1]),
    dict(type=LoadImageFromFile, key='img', channel_order='rgb'),
    dict(type=LoadImageFromFile, key='gt', channel_order='rgb'),
    dict(type=PackInputs)
]
vid4_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicFramesDataset,
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=vid4_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=1,
        pipeline=vid4_pipeline))

vid4_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicFramesDataset,
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=vid4_data_root,
        data_prefix=dict(img='BIx4', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=1,
        pipeline=vid4_pipeline))

vid4_bd_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='VID4-BDx4-Y'),
    dict(type=SSIM, convert_to='Y', prefix='VID4-BDx4-Y'),
]
vid4_bi_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='VID4-BIx4-Y'),
    dict(type=SSIM, convert_to='Y', prefix='VID4-BIx4-Y'),
]

# config for test
test_cfg = dict(type=MultiTestLoop)
test_dataloader = [
    reds_dataloader,
    vimeo_90k_bd_dataloader,
    vimeo_90k_bi_dataloader,
    udm10_dataloader,
    vid4_bd_dataloader,
    vid4_bi_dataloader,
]
test_evaluator = [
    reds_evaluator,
    vimeo_90k_bd_evaluator,
    vimeo_90k_bi_evaluator,
    udm10_evaluator,
    vid4_bd_evaluator,
    vid4_bi_evaluator,
]
