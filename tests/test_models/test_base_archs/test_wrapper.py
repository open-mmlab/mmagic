# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
from unittest import TestCase

import torch

from mmedit.registry import MODELS

test_dir = osp.join(osp.dirname(__file__), '../../..', 'tests')
config_path = osp.join(test_dir, 'configs', 'diffuser_wrapper_cfg')
model_path = osp.join(test_dir, 'configs', 'tmp_weight')


class TestWrapper(TestCase):

    def test_build(self):
        # 1. test repo_id and no_loading
        model = MODELS.build(
            dict(type='ControlNetModel', repo_id=config_path, no_loading=True))
        # test repr
        model_str = repr(model)
        self.assertIn(
            'Wrapped Module Class: <class '
            '\'diffusers.models.controlnet.ControlNetModel\'>', model_str)
        self.assertIn('Wrapped Module Name: ControlNetModel', model_str)
        self.assertIn(f'Repo ID: {config_path}', model_str)

        # 2. test save as diffuser
        model.save_pretrained(model_path)

        # 3. test repo_id and loading
        # with pretrain args
        model = MODELS.build(
            dict(
                type='ControlNetModel',
                repo_id=model_path,
                init_cfg=dict(type='Pretrained', torch_dtype=torch.float16)))
        assert all([p.dtype == torch.float16 for p in model.parameters()])

        # without pretrain args
        model = MODELS.build(dict(type='ControlNetModel', repo_id=model_path))

        shutil.rmtree(model_path)

        # 4. test loading without repo_id
        model = MODELS.build(
            dict(
                type='ControlNetModel',
                in_channels=3,
                down_block_types=['DownBlock2D'],
                block_out_channels=(20, ),
                conditioning_embedding_out_channels=(16, )), )
        model_str = repr(model)
        self.assertNotIn('Repo ID', model_str)

        # 5. test attribute error for a unknown attribute
        with self.assertRaises(AttributeError):
            model.unkonwn_attr('what\'s this?')
