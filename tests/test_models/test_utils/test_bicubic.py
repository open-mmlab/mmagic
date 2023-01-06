# Copyright (c) OpenMMLab. All rights reserved.
import scipy.io as scio
import numpy as np
import pytest
import torch

from mmedit.models.utils import imresize


def test_bicubic():
    mat = scio.loadmat('./tests/test_models/test_utils/bicubic.mat')
    img = mat['img']
    up_img = mat['resize_img']
    down_img = mat['resize_img2']

    img_torch = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    up_img_torch = imresize(img_torch, scale=4.0).squeeze(0).numpy().transpose(1, 2, 0)
    down_img_torch = imresize(img_torch, scale=1 / 4.0).squeeze(0).numpy().transpose(1, 2, 0)
    assert np.allclose(up_img, up_img_torch, atol=1e-5, rtol=1e-5)
    assert np.allclose(down_img, down_img_torch, atol=1e-5, rtol=1e-5)
