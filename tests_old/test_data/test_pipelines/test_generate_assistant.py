# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.datasets.pipelines import (GenerateCoordinateAndCell,
                                       GenerateHeatmap)


def test_generate_heatmap():
    inputs = dict(landmark=[(1, 2), (3, 4)])
    generate_heatmap = GenerateHeatmap('landmark', 4, 16)
    results = generate_heatmap(inputs)
    assert set(list(results.keys())) == set(['landmark', 'heatmap'])
    assert results['heatmap'][:, :, 0].shape == (16, 16)
    assert repr(generate_heatmap) == (
        f'{generate_heatmap.__class__.__name__}, '
        f'keypoint={generate_heatmap.keypoint}, '
        f'ori_size={generate_heatmap.ori_size}, '
        f'target_size={generate_heatmap.target_size}, '
        f'sigma={generate_heatmap.sigma}')

    generate_heatmap = GenerateHeatmap('landmark', (4, 5), (16, 17))
    results = generate_heatmap(inputs)
    assert set(list(results.keys())) == set(['landmark', 'heatmap'])
    assert results['heatmap'][:, :, 0].shape == (17, 16)


def test_generate_coordinate_and_cell():
    tensor1 = torch.randn((3, 64, 48))
    inputs1 = dict(lq=tensor1)
    coordinate1 = GenerateCoordinateAndCell(scale=3.1, target_size=(128, 96))
    results1 = coordinate1(inputs1)
    assert set(list(results1.keys())) == set(['lq', 'coord', 'cell'])
    assert repr(coordinate1) == (
        coordinate1.__class__.__name__ +
        f'sample_quantity={coordinate1.sample_quantity}, ' +
        f'scale={coordinate1.scale}, ' +
        f'target_size={coordinate1.target_size}')

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
