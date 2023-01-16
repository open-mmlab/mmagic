# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from packaging import version

from mmedit.core.evaluation.inceptions import FID, KID, InceptionV3


@pytest.mark.skipif(
    version.parse(torch.__version__) <= version.parse('1.5.1'),
    reason='torch 1.5.1 do not support interpolation with uint8')
def test_inception():
    img1 = np.random.randint(0, 256, (224, 224, 3))
    img2 = np.random.randint(0, 256, (224, 224, 3))

    # style must be either StyleGAN or pytorch
    with pytest.raises(AssertionError):
        inception = InceptionV3(style='some')

    # for StyleGAN style inception
    inception = InceptionV3(style='StyleGAN')

    t = inception.img2tensor(img1)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.uint8
    assert t.shape == (1, 3, 224, 224)

    t = inception.forward_inception(t)
    assert t.device == torch.device('cpu')
    assert t.shape == (1, 2048)

    # for PyTorch style inception
    inception = InceptionV3(style='pytorch')

    t = inception.img2tensor(img1)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert t.shape == (1, 3, 224, 224)
    assert np.logical_and(0 <= t, t <= 1).all()

    t = inception.forward_inception(t)
    assert t.device == torch.device('cpu')
    assert t.shape == (1, 2048)

    # test `__call__` method in cpu
    inception = InceptionV3(device='cpu')
    feats = inception(img1, img2)
    assert isinstance(feats, tuple) and len(feats) == 2
    assert feats[0].shape == (1, 2048)
    assert feats[1].shape == (1, 2048)

    # test `__call__` method in cuda
    if torch.cuda.is_available():
        inception = InceptionV3(device='cuda')
        feats = inception(img1, img2)
        assert isinstance(feats, tuple) and len(feats) == 2
        assert feats[0].shape == (1, 2048)
        assert feats[1].shape == (1, 2048)


def test_fid():
    fid = FID()
    fid_result = fid(np.ones((10, 2048)), np.ones((10, 2048)))
    assert isinstance(fid_result, float)
    assert fid_result == 0.0


def test_kid():
    kid = KID(num_repeats=1, sample_size=10)
    kid_result = kid(np.ones((10, 2048)), np.ones((10, 2048)))
    assert isinstance(kid_result, dict)
    assert 'KID_MEAN' in kid_result and 'KID_STD' in kid_result
    assert kid_result['KID_MEAN'] == 0.0
    assert kid_result['KID_STD'] == 0.0

    # if sample size > number of samples
    with pytest.raises(ValueError):
        kid = KID(sample_size=100)
        kid_result = kid(np.ones((10, 2048)), np.ones((10, 2048)))
