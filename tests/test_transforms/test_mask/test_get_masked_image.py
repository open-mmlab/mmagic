# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmedit.transforms import GetMaskedImage


def test_masked_img():
    img = np.random.rand(4, 4, 1).astype(np.float32)
    mask = np.zeros((4, 4, 1), dtype=np.float32)
    mask[1, 1] = 1

    results = dict(gt_img=img, mask=mask)
    get_masked_img = GetMaskedImage()
    results = get_masked_img(results)
    masked_img = img * (1. - mask)
    assert np.array_equal(results['masked_img'], masked_img)

    name_ = repr(get_masked_img)
    class_name = get_masked_img.__class__.__name__
    assert name_ == class_name + "(img_name='gt_img', mask_name='mask')"
