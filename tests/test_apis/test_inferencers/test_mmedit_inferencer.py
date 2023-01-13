# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmedit.apis.inferencers.mmedit_inferencer import MMEditInferencer
from mmedit.utils import register_all_modules

register_all_modules()


def test_mmedit_inferencer():
    with pytest.raises(Exception) as e_info:
        inferencer_instance = MMEditInferencer('unconditional', ['error_type'],
                                               None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = MMEditInferencer('matting', ['error_type'], None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = MMEditInferencer('inpainting', ['error_type'],
                                               None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = MMEditInferencer('translation', ['error_type'],
                                               None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = MMEditInferencer('restoration', ['error_type'],
                                               None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = MMEditInferencer('video_restoration',
                                               ['error_type'], None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = MMEditInferencer('video_interpolation',
                                               ['error_type'], None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = MMEditInferencer('dog', ['error_type'], None)

    print(e_info)

    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'sngan_proj',
        'sngan-proj_woReLUinplace_lr2e-4-ndisc5-1xb64_cifar10-32x32.py')
    inferencer_instance = \
        MMEditInferencer('conditional',
                         cfg,
                         None,
                         extra_parameters={'sample_model': 'orig'})
    inference_result = inferencer_instance(label=1)
    result_img = inference_result[1]
    assert result_img.shape == (4, 3, 32, 32)

    extra_parameters = inferencer_instance.get_extra_parameters()
    assert len(extra_parameters) == 2
