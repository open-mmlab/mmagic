import numpy as np
import pytest
import torch
from mmedit.core import tensor2img
from torchvision.utils import make_grid


def test_tensor2img():
    tensor_4d = torch.FloatTensor(2, 3, 4, 4).uniform_(0, 1)
    tensor_3d_1 = torch.FloatTensor(3, 4, 4).uniform_(0, 1)
    tensor_3d_2 = torch.FloatTensor(3, 6, 6).uniform_(0, 1)
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
    rlt = tensor2img(tensor_4d, out_type=np.uint8, min_max=(0, 1))
    tensor_4d_np = make_grid(tensor_4d, nrow=1, normalize=False).numpy()
    tensor_4d_np = np.transpose(tensor_4d_np[[2, 1, 0], :, :], (1, 2, 0))
    np.testing.assert_almost_equal(rlt, (tensor_4d_np * 255).round())

    # 3d
    rlt = tensor2img([tensor_3d_1, tensor_3d_2],
                     out_type=np.uint8,
                     min_max=(0, 1))
    tensor_3d_1_np = tensor_3d_1.numpy()
    tensor_3d_1_np = np.transpose(tensor_3d_1_np[[2, 1, 0], :, :], (1, 2, 0))
    tensor_3d_2_np = tensor_3d_2.numpy()
    tensor_3d_2_np = np.transpose(tensor_3d_2_np[[2, 1, 0], :, :], (1, 2, 0))
    np.testing.assert_almost_equal(rlt[0], (tensor_3d_1_np * 255).round())
    np.testing.assert_almost_equal(rlt[1], (tensor_3d_2_np * 255).round())

    # 2d
    rlt = tensor2img(tensor_2d, out_type=np.uint8, min_max=(0, 1))
    tensor_2d_np = tensor_2d.numpy()
    np.testing.assert_almost_equal(rlt, (tensor_2d_np * 255).round())
    rlt = tensor2img(tensor_2d, out_type=np.float32, min_max=(0, 1))
    np.testing.assert_almost_equal(rlt, tensor_2d_np)

    rlt = tensor2img(tensor_2d, out_type=np.float32, min_max=(0.1, 0.5))
    tensor_2d_np = (np.clip(tensor_2d_np, 0.1, 0.5) - 0.1) / 0.4
    np.testing.assert_almost_equal(rlt, tensor_2d_np)
