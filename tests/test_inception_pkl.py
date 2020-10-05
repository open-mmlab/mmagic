import pytest
import torch

from mmedit.models.backbones import InceptionV3


def test_inceptionv3():

    # sanity check for construction
    with pytest.raises(AssertionError):
        inception = InceptionV3([100])

    inception = InceptionV3()
    # sanity check for standard forward
    input_x = torch.randn((2, 3, 256, 256))
    res = inception(input_x)
    assert len(res) == len(inception.output_blocks)
    assert tuple(res[0].size()) == (2, 2048, 1, 1)

    # cuda testing
    if torch.cuda.is_available():
        input_x = input_x.cuda()
        inception.cuda()
        res = inception(input_x)
        assert len(res) == len(inception.output_blocks)
        assert tuple(res[0].size()) == (2, 2048, 1, 1)
