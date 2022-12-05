# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors import Restormer


def test_restormer_cpu():
    """Test Restormer."""

    # Image Deblurring: Motion Deblurring or Image Deraining
    net = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False)
    img = torch.rand(1, 3, 64, 64)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 64, 64)

    # Image Denoising Gray
    net = Restormer(
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 64, 64)

    # Image Denoising Color
    net = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 64, 64)

    # Image Dual Defocus Deblurring
    net = Restormer(
        inp_channels=6,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=True)
    img = torch.rand(1, 6, 64, 64)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 64, 64)


def test_swinir_cuda():
    net = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False)
    img = torch.rand(1, 3, 64, 64)

    # x4 model lightweight SR forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 3, 64, 64)
