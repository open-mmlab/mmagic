# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors import SwinIRNet


def test_swinir_cpu():
    """Test SwinIRNet."""

    # x2 model classical SR
    net = SwinIRNet(
        upscale=2,
        in_channels=3,
        img_size=48,
        window_size=8,
        img_range=1.0,
        depths=[6],
        embed_dim=60,
        num_heads=[6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv')

    # x3 model classical SR, initialization and forward (cpu)
    net = SwinIRNet(
        upscale=3,
        in_channels=3,
        img_size=16,
        window_size=8,
        img_range=1.0,
        depths=[2],
        embed_dim=8,
        num_heads=[2],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv')
    img = torch.rand(1, 3, 16, 16)
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 48, 48)

    # x4 model lightweight SR, initialization and forward (cpu)
    net = SwinIRNet(
        upscale=4,
        in_channels=3,
        img_size=16,
        window_size=8,
        img_range=1.0,
        depths=[2],
        embed_dim=8,
        num_heads=[2],
        mlp_ratio=2,
        upsampler='pixelshuffledirect',
        resi_connection='1conv')
    output = net(img)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 64, 64)


def test_swinir_cuda():
    net = SwinIRNet(
        upscale=4,
        in_channels=3,
        img_size=16,
        window_size=8,
        img_range=1.0,
        depths=[2],
        embed_dim=8,
        num_heads=[2],
        mlp_ratio=2,
        upsampler='pixelshuffledirect',
        resi_connection='1conv')
    img = torch.rand(1, 3, 16, 16)

    # x4 model lightweight SR forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 3, 64, 64)
