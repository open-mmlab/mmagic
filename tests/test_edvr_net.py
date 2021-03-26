import pytest
import torch

from mmedit.models.backbones.sr_backbones.edvr_net import (EDVRNet,
                                                           PCDAlignment,
                                                           TSAFusion)


def test_pcd_alignment():
    """Test PCDAlignment."""

    # gpu (since it has dcn, only supports gpu testing)
    if torch.cuda.is_available():
        pcd_alignment = PCDAlignment(mid_channels=4, deform_groups=2)
        input_list = []
        for i in range(3, 0, -1):
            input_list.append(torch.rand(1, 4, 2**i, 2**i))

        pcd_alignment = pcd_alignment.cuda()
        input_list = [v.cuda() for v in input_list]
        output = pcd_alignment(input_list, input_list)
        assert output.shape == (1, 4, 8, 8)

        with pytest.raises(AssertionError):
            pcd_alignment(input_list[0:2], input_list)


def test_tsa_fusion():
    """Test TSAFusion."""

    # cpu
    tsa_fusion = TSAFusion(mid_channels=4, num_frames=5, center_frame_idx=2)
    input_tensor = torch.rand(1, 5, 4, 8, 8)

    output = tsa_fusion(input_tensor)
    assert output.shape == (1, 4, 8, 8)

    # gpu
    if torch.cuda.is_available():
        tsa_fusion = tsa_fusion.cuda()
        input_tensor = input_tensor.cuda()
        output = tsa_fusion(input_tensor)
        assert output.shape == (1, 4, 8, 8)


def test_edvrnet():
    """Test EDVRNet."""

    # gpu
    if torch.cuda.is_available():
        # with tsa
        edvrnet = EDVRNet(
            3,
            3,
            mid_channels=8,
            num_frames=5,
            deform_groups=2,
            num_blocks_extraction=1,
            num_blocks_reconstruction=1,
            center_frame_idx=2,
            with_tsa=True).cuda()
        input_tensor = torch.rand(1, 5, 3, 8, 8).cuda()
        edvrnet.init_weights(pretrained=None)
        output = edvrnet(input_tensor)
        assert output.shape == (1, 3, 32, 32)

        # without tsa
        edvrnet = EDVRNet(
            3,
            3,
            mid_channels=8,
            num_frames=5,
            deform_groups=2,
            num_blocks_extraction=1,
            num_blocks_reconstruction=1,
            center_frame_idx=2,
            with_tsa=False).cuda()

        output = edvrnet(input_tensor)
        assert output.shape == (1, 3, 32, 32)

        with pytest.raises(AssertionError):
            # The height and width of inputs should be a multiple of 4
            input_tensor = torch.rand(1, 5, 3, 3, 3).cuda()
            edvrnet(input_tensor)

        with pytest.raises(TypeError):
            # pretrained should be str or None
            edvrnet.init_weights(pretrained=[1])
