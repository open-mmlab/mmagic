import pytest
import torch

from mmedit.models.backbones.sr_backbones.basicvsr import BasicVSR


def test_basicvsr():
    """Test BasicVSR."""

    # cpu
    basicvsr = BasicVSR(mid_channels=64, num_blocks=30, spynet_pretrained=None)
    input_tensor = torch.rand(1, 5, 3, 64, 64)
    basicvsr.init_weights(pretrained=None)
    output = basicvsr(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)

    # gpu
    if torch.cuda.is_available():
        basicvsr = BasicVSR(
            mid_channels=64, num_blocks=30, spynet_pretrained=None).cuda()
        input_tensor = torch.rand(1, 5, 3, 64, 64).cuda()
        basicvsr.init_weights(pretrained=None)
        output = basicvsr(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)

    with pytest.raises(AssertionError):
        # The height and width of inputs should be at least 64
        input_tensor = torch.rand(1, 5, 3, 61, 61)
        basicvsr(input_tensor)

    with pytest.raises(TypeError):
        # pretrained should be str or None
        basicvsr.init_weights(pretrained=[1])
