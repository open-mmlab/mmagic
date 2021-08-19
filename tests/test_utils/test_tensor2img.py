# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from torchvision.utils import make_grid

from mmedit.core import tensor2img


def test_tensor2img():
    tensor_4d_1 = torch.FloatTensor(2, 3, 4, 4).uniform_(0, 1)
    tensor_4d_2 = torch.FloatTensor(1, 3, 4, 4).uniform_(0, 1)
    tensor_4d_3 = torch.FloatTensor(3, 1, 4, 4).uniform_(0, 1)
    tensor_4d_4 = torch.FloatTensor(1, 1, 4, 4).uniform_(0, 1)
    tensor_3d_1 = torch.FloatTensor(3, 4, 4).uniform_(0, 1)
    tensor_3d_2 = torch.FloatTensor(3, 6, 6).uniform_(0, 1)
    tensor_3d_3 = torch.FloatTensor(1, 6, 6).uniform_(0, 1)
    tensor_2d = torch.FloatTensor(4, 4).uniform_(0, 1)

    with pytest.raises(TypeError):
        # input is not a tensor
        tensor2img(4)
    with pytest.raises(TypeError):
        # input is not a list of tensors
        tensor2img([tensor_3d_1, 4])
    with pytest.raises(ValueError):
        # unsupported 5D tensor
        tensor2img(torch.FloatTensor(2, 2, 3, 4, 4).uniform_(0, 1))

    # 4d
    rlt = tensor2img(tensor_4d_1, out_type=np.uint8, min_max=(0, 1))
    assert rlt.dtype == np.uint8
    tensor_4d_1_np = make_grid(tensor_4d_1, nrow=1, normalize=False).numpy()
    tensor_4d_1_np = np.transpose(tensor_4d_1_np[[2, 1, 0], :, :], (1, 2, 0))
    np.testing.assert_almost_equal(rlt, (tensor_4d_1_np * 255).round())

    rlt = tensor2img(tensor_4d_2, out_type=np.uint8, min_max=(0, 1))
    assert rlt.dtype == np.uint8
    tensor_4d_2_np = tensor_4d_2.squeeze().numpy()
    tensor_4d_2_np = np.transpose(tensor_4d_2_np[[2, 1, 0], :, :], (1, 2, 0))
    np.testing.assert_almost_equal(rlt, (tensor_4d_2_np * 255).round())

    rlt = tensor2img(tensor_4d_3, out_type=np.uint8, min_max=(0, 1))
    assert rlt.dtype == np.uint8
    tensor_4d_3_np = make_grid(tensor_4d_3, nrow=1, normalize=False).numpy()
    tensor_4d_3_np = np.transpose(tensor_4d_3_np[[2, 1, 0], :, :], (1, 2, 0))
    np.testing.assert_almost_equal(rlt, (tensor_4d_3_np * 255).round())

    rlt = tensor2img(tensor_4d_4, out_type=np.uint8, min_max=(0, 1))
    assert rlt.dtype == np.uint8
    tensor_4d_4_np = tensor_4d_4.squeeze().numpy()
    np.testing.assert_almost_equal(rlt, (tensor_4d_4_np * 255).round())

    # 3d
    rlt = tensor2img([tensor_3d_1, tensor_3d_2],
                     out_type=np.uint8,
                     min_max=(0, 1))
    assert rlt[0].dtype == np.uint8
    tensor_3d_1_np = tensor_3d_1.numpy()
    tensor_3d_1_np = np.transpose(tensor_3d_1_np[[2, 1, 0], :, :], (1, 2, 0))
    tensor_3d_2_np = tensor_3d_2.numpy()
    tensor_3d_2_np = np.transpose(tensor_3d_2_np[[2, 1, 0], :, :], (1, 2, 0))
    np.testing.assert_almost_equal(rlt[0], (tensor_3d_1_np * 255).round())
    np.testing.assert_almost_equal(rlt[1], (tensor_3d_2_np * 255).round())

    rlt = tensor2img(tensor_3d_3, out_type=np.uint8, min_max=(0, 1))
    assert rlt.dtype == np.uint8
    tensor_3d_3_np = tensor_3d_3.squeeze().numpy()
    np.testing.assert_almost_equal(rlt, (tensor_3d_3_np * 255).round())

    # 2d
    rlt = tensor2img(tensor_2d, out_type=np.uint8, min_max=(0, 1))
    assert rlt.dtype == np.uint8
    tensor_2d_np = tensor_2d.numpy()
    np.testing.assert_almost_equal(rlt, (tensor_2d_np * 255).round())
    rlt = tensor2img(tensor_2d, out_type=np.float32, min_max=(0, 1))
    assert rlt.dtype == np.float32
    np.testing.assert_almost_equal(rlt, tensor_2d_np)

    rlt = tensor2img(tensor_2d, out_type=np.float32, min_max=(0.1, 0.5))
    assert rlt.dtype == np.float32
    tensor_2d_np = (np.clip(tensor_2d_np, 0.1, 0.5) - 0.1) / 0.4
    np.testing.assert_almost_equal(rlt, tensor_2d_np)
