# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmedit.datasets.pipelines import MATLABResize


def test_matlab_resize():
    results = {}

    # give scale
    results['lq'] = np.ones((16, 16, 3))
    imresize = MATLABResize(keys=['lq'], scale=0.25)
    results = imresize(results)
    assert results['lq'].shape == (4, 4, 3)

    # give scale
    results['lq'] = np.ones((16, 16, 3))
    imresize = MATLABResize(keys=['lq'], output_shape=(6, 6))
    results = imresize(results)
    assert results['lq'].shape == (6, 6, 3)

    # kernel must equal 'bicubic'
    with pytest.raises(ValueError):
        MATLABResize(keys=['lq'], kernel='abc')

    # kernel_width must equal 4.0
    with pytest.raises(ValueError):
        MATLABResize(keys=['lq'], kernel_width=10)

    # scale and output_shape cannot be both None
    with pytest.raises(ValueError):
        MATLABResize(keys=['lq'])

    assert repr(imresize) == imresize.__class__.__name__ \
        + "(keys=['lq'], scale=None, output_shape=(6, 6), " \
        + 'kernel=bicubic, kernel_width=4.0)'
