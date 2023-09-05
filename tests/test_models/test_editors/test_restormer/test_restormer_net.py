# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors import Restormer


@pytest.mark.skipif(
    torch.__version__ < '1.8.0',
    reason='skip on torch<1.8 due to unsupported PixelUnShuffle')
def test_restormer_cpu():
    """Test Restormer."""

    # Motion Deblurring or Image Deraining
    net = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=24,
        num_blocks=[2, 2, 2, 4],
        num_refinement_blocks=1,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False)
    img = torch.rand(1, 3, 16, 16)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 16, 16)

    # Image Denoising Gray
    net = Restormer(
        inp_channels=1,
        out_channels=1,
        dim=24,
        num_blocks=[2, 2, 2, 4],
        num_refinement_blocks=1,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False)
    img = torch.rand(1, 1, 16, 16)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1, 16, 16)

    # Image Denoising Color
    net = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=24,
        num_blocks=[2, 2, 2, 4],
        num_refinement_blocks=1,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False)
    img = torch.rand(1, 3, 16, 16)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 16, 16)

    # Image Dual Defocus Deblurring
    net = Restormer(
        inp_channels=6,
        out_channels=3,
        dim=24,
        num_blocks=[2, 2, 2, 4],
        num_refinement_blocks=1,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=True,
        dual_keys=['imgL', 'imgR'])
    img = dict()
    img['imgL'] = torch.rand(1, 3, 16, 16)
    img['imgR'] = torch.rand(1, 3, 16, 16)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 16, 16)


@pytest.mark.skipif(
    torch.__version__ < '1.8.0',
    reason='skip on torch<1.8 due to unsupported PixelUnShuffle')
def test_restormer_cuda():
    net = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=24,
        num_blocks=[2, 2, 2, 4],
        num_refinement_blocks=1,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False)
    img = torch.rand(1, 3, 16, 16)

    # Image Deblurring or Image Deraining (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 3, 16, 16)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
