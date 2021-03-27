import torch

from mmedit.datasets.pipelines import GenerateCoordinateAndCell


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
