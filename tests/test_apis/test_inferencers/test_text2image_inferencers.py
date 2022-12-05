# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import torch
import pytest

from mmedit.apis.inferencers.text2image_inferencer import \
    Text2ImageInferencer
from mmedit.utils import register_all_modules

register_all_modules()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
def test_translation_inferencer():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'disco_diffusion',
        'disco-diffusion_adm-u-finetuned_imagenet-512x512.py')
    text={0:['sad']}
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data', 'disco_result.png')

    inferencer_instance = \
        Text2ImageInferencer(cfg, None, extra_parameters={'num_inference_steps':2})
    inferencer_instance(text=text)
    inference_result = inferencer_instance(
        text=text, result_out_dir=result_out_dir)
    result_img = inference_result[1]
    assert result_img[0].cpu().numpy().shape == (3, 512, 512)


if __name__ == '__main__':
    test_translation_inferencer()
