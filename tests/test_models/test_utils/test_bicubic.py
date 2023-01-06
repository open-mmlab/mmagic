# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import scipy.io as scio
import torch

from mmedit.models.utils import imresize


def test_bicubic():
    mat = scio.loadmat('./tests/test_models/test_utils/bicubic.mat')
    img = mat['img']
    up_img = mat['resize_img']
    down_img = mat['resize_img2']

    batch_size = 4
    img_torch = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).repeat(
        batch_size, 1, 1, 1)
    for i in range(batch_size):
        up_img_torch = imresize(
            img_torch, scale=4.0)[i, ...].numpy().transpose(1, 2, 0)
        down_img_torch = imresize(
            img_torch, scale=1 / 4.0)[i, ...].numpy().transpose(1, 2, 0)
        assert np.allclose(up_img, up_img_torch, atol=1e-5, rtol=1e-5)
        assert np.allclose(down_img, down_img_torch, atol=1e-5, rtol=1e-5)
