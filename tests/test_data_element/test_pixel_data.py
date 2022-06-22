# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmedit.data_element import PixelData


def test_pixel_data():

    img_data = dict(
        img=np.random.randint(0, 255, (3, 256, 256)),
        tensor=torch.rand((3, 256, 256)))

    gt_img = PixelData(**img_data)

    assert (gt_img.img == img_data['img']).all()
    assert (gt_img.tensor == img_data['tensor']).all()
    assert isinstance(gt_img.img, np.ndarray)
    assert isinstance(gt_img.tensor, torch.Tensor)

    with pytest.raises(AttributeError):
        # reserved private key name
        PixelData(_metainfo_fields='')

    with pytest.raises(AssertionError):
        # allow tensor or numpy array only
        PixelData(img='')

    with pytest.raises(AssertionError):
        # size not match
        img_data = dict(
            img=np.random.randint(0, 255, (3, 256, 256)),
            tensor=torch.rand((3, 256, 257)))
        gt_img = PixelData(**img_data)

    with pytest.raises(AssertionError):
        # only 2,3,4 dim
        img_data = dict(tensor=torch.rand((3, 3, 3, 256, 256)))
        gt_img = PixelData(**img_data)
