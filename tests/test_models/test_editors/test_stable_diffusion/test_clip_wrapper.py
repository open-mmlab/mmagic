# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from transformers import CLIPConfig

from mmedit.models.editors.stable_diffusion.clip_wrapper import (
    StableDiffusionSafetyChecker, load_clip_submodels)


def test_clip_wrapper():
    clipconfig = CLIPConfig()
    safety_checker = StableDiffusionSafetyChecker(clipconfig)

    clip_input = torch.rand((1, 3, 224, 224))
    images_input = torch.rand((1, 512, 512, 3))

    result = safety_checker.forward(clip_input, images_input)
    assert result[0].shape == (1, 512, 512, 3)


def test_load_clip_submodels():
    tokenizer_path = dict(
        subdir_name='tokenizer',
        merges='1.txt',
        special_tokens_map='1.json',
        tokenizer_config='1.json',
        vocab='1.json')

    text_encoder_path = dict(
        subdir_name='text_encoder', config='1.json', pytorch_model='1.bin')

    feature_extractor_path = dict(
        subdir_name='feature_extractor', config='1.json')

    safety_checker_path = dict(
        subdir_name='safety_checker', config='1.json', pytorch_model='1.bin')

    pretrained_ckpt_path = {}
    pretrained_ckpt_path['tokenizer'] = tokenizer_path
    pretrained_ckpt_path['feature_extractor'] = None
    pretrained_ckpt_path['text_encoder'] = None
    pretrained_ckpt_path['safety_checker'] = None

    submodels = []
    with pytest.raises(Exception):
        load_clip_submodels(pretrained_ckpt_path, submodels, True)

    pretrained_ckpt_path = {}
    pretrained_ckpt_path['tokenizer'] = None
    pretrained_ckpt_path['feature_extractor'] = feature_extractor_path
    pretrained_ckpt_path['text_encoder'] = None
    pretrained_ckpt_path['safety_checker'] = None
    submodels = []
    with pytest.raises(Exception):
        load_clip_submodels(pretrained_ckpt_path, submodels, True)

    pretrained_ckpt_path = {}
    pretrained_ckpt_path['tokenizer'] = None
    pretrained_ckpt_path['feature_extractor'] = None
    pretrained_ckpt_path['text_encoder'] = text_encoder_path
    pretrained_ckpt_path['safety_checker'] = None
    submodels = []
    with pytest.raises(Exception):
        load_clip_submodels(pretrained_ckpt_path, submodels, True)

    pretrained_ckpt_path = {}
    pretrained_ckpt_path['tokenizer'] = None
    pretrained_ckpt_path['feature_extractor'] = None
    pretrained_ckpt_path['text_encoder'] = None
    pretrained_ckpt_path['safety_checker'] = safety_checker_path
    submodels = []
    with pytest.raises(Exception):
        load_clip_submodels(pretrained_ckpt_path, submodels, True)


if __name__ == '__main__':
    test_load_clip_submodels()
