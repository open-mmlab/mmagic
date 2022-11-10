# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmedit.apis.inferencers.mmedit_inferencer import MMEditInferencer
from mmedit.utils import register_all_modules

register_all_modules()


def test_mmedit_inferencer():
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


if __name__ == '__main__':
    test_mmedit_inferencer()
