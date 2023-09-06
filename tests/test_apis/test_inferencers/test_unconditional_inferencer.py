# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch

from mmagic.apis.inferencers.unconditional_inferencer import \
    UnconditionalInferencer
from mmagic.utils import register_all_modules

register_all_modules()


def test_unconditional_inferencer():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'styleganv1',
        'styleganv1_ffhq-256x256_8xb4-25Mimgs.py')
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data/out',
        'unconditional_result.png')

    inferencer_instance = \
        UnconditionalInferencer(cfg,
                                None,
                                extra_parameters={
                                    'num_batches': 1,
                                    'sample_model': 'orig'})
    # test no result_out_dir
    inferencer_instance()
    inference_result = inferencer_instance(result_out_dir=result_out_dir)
    result_img = inference_result[1]
    assert result_img.detach().numpy().shape == (1, 3, 256, 256)

    # test additional input for extra parameters
    noise = torch.randn(1, 512)
    sample_kwargs = sample_kwargs = {
        'return_noise': True,
        'input_is_latent': False
    }

    inferencer_instance = \
        UnconditionalInferencer(cfg,
                                None,
                                extra_parameters={
                                    'num_batches': 1,
                                    'sample_model': 'orig',
                                    'noise': noise,
                                    'sample_kwargs': sample_kwargs})
    # test no result_out_dir
    inferencer_instance()
    inference_result = inferencer_instance(result_out_dir=result_out_dir)
    result_img = inference_result[1]
    assert result_img.detach().numpy().shape == (1, 3, 256, 256)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
