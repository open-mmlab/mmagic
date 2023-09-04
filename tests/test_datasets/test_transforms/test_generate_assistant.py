# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.datasets.transforms import (GenerateCoordinateAndCell,
                                        GenerateFacialHeatmap,
                                        LoadImageFromFile)


def test_generate_coordinate_and_cell():

    tensor1 = torch.randn((3, 64, 48))
    inputs1 = dict(lq=tensor1)
    coordinate1 = GenerateCoordinateAndCell(scale=3.1, target_size=(128, 96))
    results1 = coordinate1(inputs1)
    assert set(list(results1.keys())) == set(['lq', 'coord', 'cell'])
    assert repr(coordinate1) == (
        coordinate1.__class__.__name__ +
        f'(sample_quantity={coordinate1.sample_quantity}, ' +
        f'scale={coordinate1.scale}, ' +
        f'target_size={coordinate1.target_size}, ' +
        f'reshape_gt={coordinate1.reshape_gt})')

    tensor2 = torch.randn((3, 64, 48))
    inputs2 = dict(gt=tensor2)
    coordinate2 = GenerateCoordinateAndCell(
        sample_quantity=64 * 48, scale=3.1, target_size=(128, 96))
    results2 = coordinate2(inputs2)
    assert set(list(results2.keys())) == set(['gt', 'coord', 'cell'])
    assert results2['gt'].shape == (64 * 48, 3)

    inputs3 = dict()
    coordinate3 = GenerateCoordinateAndCell(
        sample_quantity=64 * 48, scale=3.1, target_size=(128, 96))
    results3 = coordinate3(inputs3)
    assert set(list(results3.keys())) == set(['coord', 'cell'])


def test_generate_facial_heatmap():

    results = dict(gt_path='tests/data/image/face/000001.png', key='000001')
    loader = LoadImageFromFile(key='gt', channel_order='rgb')
    results = loader(results)

    generate_heatmap = GenerateFacialHeatmap('gt', 256, 32)
    results1 = generate_heatmap(results)
    results1 = generate_heatmap(results)
    assert set(list(results1.keys())) == set([
        'gt_path', 'gt', 'ori_gt_shape', 'gt_heatmap', 'key',
        'gt_channel_order', 'gt_color_type'
    ])
    assert results1['gt_heatmap'].shape == (32, 32, 68)

    generate_heatmap = GenerateFacialHeatmap('gt', (256, 256), (32, 64))
    results2 = generate_heatmap(results)
    assert set(list(results2.keys())) == set([
        'gt_path', 'gt', 'ori_gt_shape', 'gt_heatmap', 'key',
        'gt_channel_order', 'gt_color_type'
    ])
    assert results2['gt_heatmap'].shape == (32, 64, 68)

    generate_heatmap = GenerateFacialHeatmap(
        'gt', (256, 256), (32, 64), use_cache=False)
    results2 = generate_heatmap(results)
    assert set(list(results2.keys())) == set([
        'gt_path', 'gt', 'ori_gt_shape', 'gt_heatmap', 'key',
        'gt_channel_order', 'gt_color_type'
    ])
    assert results2['gt_heatmap'].shape == (32, 64, 68)

    assert repr(generate_heatmap) == (
        generate_heatmap.__class__.__name__ +
        f'(image_key={generate_heatmap.image_key}, ' +
        f'ori_size={generate_heatmap.ori_size}, ' +
        f'target_size={generate_heatmap.target_size}, ' +
        f'sigma={generate_heatmap.sigma}, '
        f'use_cache={generate_heatmap.use_cache})')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
