# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import shutil
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

test_dir = osp.join(osp.dirname(__file__), '..', '..', '..', 'tests')
config_path = osp.join(test_dir, 'configs', 'diffuser_wrapper_cfg')
model_path = osp.join(test_dir, 'configs', 'tmp_weight')
ckpt_path = osp.join(test_dir, 'configs', 'ckpt')

register_all_modules()


class TestWrapper(TestCase):

    def test_build(self):
        # mock SiLU
        if digit_version(TORCH_VERSION) <= digit_version('1.6.0'):
            from mmagic.models.editors.ddpm.denoising_unet import SiLU
            torch.nn.SiLU = SiLU

        # 1. test from config
        model = MODELS.build(
            dict(type='ControlNetModel', from_config=config_path))
        model.init_weights()  # test init_weights without warning
        model_str = repr(model)
        self.assertIn(
            'Wrapped Module Class: <class '
            '\'diffusers.models.controlnet.ControlNetModel\'>', model_str)
        self.assertIn('Wrapped Module Name: ControlNetModel', model_str)
        self.assertIn(f'From Config: {config_path}', model_str)

        # 2. test save as diffuser
        os.makedirs(model_path, exist_ok=True)
        if digit_version(TORCH_VERSION) < digit_version('2.0.1'):
            model.save_pretrained(model_path, safe_serialization=False)
        else:
            model.save_pretrained(model_path)

        # 3. test from_pretrained
        model = MODELS.build(
            dict(
                type='ControlNetModel',
                from_pretrained=model_path,
                torch_dtype=torch.float16))
        assert all([p.dtype == torch.float16 for p in model.parameters()])
        model_str = repr(model)
        self.assertIn(f'From Pretrained: {model_path}', model_str)

        # save ckpt to test init_weights
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(model.state_dict(), osp.join(ckpt_path, 'model.pth'))

        # test raise warning when init_cfg is passed
        model = MODELS.build(
            dict(
                type='ControlNetModel',
                from_pretrained=model_path,
                torch_dtype=torch.float16,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint=osp.join(ckpt_path, 'model.pth'))))
        model.init_weights()

        # delete saved model to save space
        if 'win' not in platform.system().lower():
            shutil.rmtree(model_path)
            shutil.rmtree(ckpt_path)

        # 4. test loading without repo_id
        model = MODELS.build(
            dict(
                type='ControlNetModel',
                in_channels=3,
                down_block_types=['DownBlock2D'],
                block_out_channels=(32, ),
                cross_attention_dim=16,
                attention_head_dim=2,
                conditioning_embedding_out_channels=(16, )), )
        model_str = repr(model)
        self.assertNotIn('From Config:', model_str)
        self.assertNotIn('From Pretrained:', model_str)

        # 5. test attribute error for a unknown attribute
        with self.assertRaises(AttributeError):
            model.unkonwn_attr('what\'s this?')

        # 6. test init_weights
        model.init_weights()

        # 7. test forward function
        forward_mock = MagicMock()
        model.model.forward = forward_mock
        model(**dict(t='t', control='control'))
        _, called_kwargs = forward_mock.call_args
        self.assertEqual(called_kwargs['t'], 't')
        self.assertEqual(called_kwargs['control'], 'control')

        # 8. test other attribute share with BaseModule and model
        register_buffer_mock = MagicMock()
        model.model.registrer_buffer = register_buffer_mock
        model.registrer_buffer('buffer', 123)
        called_args, _ = register_buffer_mock.call_args
        self.assertEqual(called_args, ('buffer', 123))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
