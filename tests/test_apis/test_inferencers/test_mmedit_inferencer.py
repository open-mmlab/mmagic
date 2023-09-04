# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmagic.apis.inferencers import Inferencers
from mmagic.utils import register_all_modules

register_all_modules()


def test_mmagic_inferencer():
    with pytest.raises(Exception) as e_info:
        inferencer_instance = Inferencers('colorization', ['error_type'], None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = Inferencers('unconditional', ['error_type'],
                                          None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = Inferencers('matting', ['error_type'], None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = Inferencers('inpainting', ['error_type'], None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = Inferencers('translation', ['error_type'], None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = Inferencers('restoration', ['error_type'], None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = Inferencers('video_restoration', ['error_type'],
                                          None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = Inferencers('video_interpolation',
                                          ['error_type'], None)

    with pytest.raises(Exception) as e_info:
        inferencer_instance = Inferencers('dog', ['error_type'], None)

    print(e_info)

    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'sngan_proj',
        'sngan-proj_woReLUinplace_lr2e-4-ndisc5-1xb64_cifar10-32x32.py')
    inferencer_instance = \
        Inferencers('conditional',
                    cfg,
                    None,
                    extra_parameters={'sample_model': 'orig'})
    inference_result = inferencer_instance(label=1)
    result_img = inference_result[1]
    assert result_img.shape == (4, 3, 32, 32)

    extra_parameters = inferencer_instance.get_extra_parameters()
    assert len(extra_parameters) == 2


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
