# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine.model import MMDistributedDataParallel

from mmagic.models import BaseTranslationModel


class ToyTranslationModel(BaseTranslationModel):

    def __init__(self, generator, discriminator=None):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            default_domain='A',
            reachable_domains=['A'],
            related_domains=['A', 'B'],
            data_preprocessor=None)


class TestBaseTranslationModel(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = dict(
            type='ResnetGenerator',
            in_channels=3,
            out_channels=3,
            base_channels=64,
            norm_cfg=dict(type='IN'),
            use_dropout=False,
            num_blocks=9,
            padding_mode='reflect',
            init_cfg=dict(type='normal', gain=0.02))
        cls.discriminator = dict(
            type='PatchDiscriminator',
            in_channels=3,
            base_channels=64,
            num_conv=3,
            norm_cfg=dict(type='IN'),
            init_cfg=dict(type='normal', gain=0.02))

    def test_init(self):
        # test disc is None
        model = ToyTranslationModel(
            generator=self.generator, discriminator=None)
        self.assertIsNone(model.discriminators)

        # test disc is not None
        model = ToyTranslationModel(
            generator=self.generator, discriminator=self.discriminator)
        self.assertIsNotNone(model.discriminators)

    def test_get_module(self):
        generator_mock = MagicMock(spec=MMDistributedDataParallel)
        generator_mock_module = MagicMock()
        generator_mock.module = generator_mock_module

        model = ToyTranslationModel(
            generator=self.generator, discriminator=None)
        module = model.get_module(generator_mock)
        self.assertEqual(module, generator_mock_module)

    def test_forward(self):
        model = ToyTranslationModel(
            generator=self.generator, discriminator=None)
        res = model.forward_test(
            img=torch.randn(1, 3, 64, 64), target_domain=None)
        self.assertEqual(res['target'].shape, (1, 3, 64, 64))
        self.assertEqual(res['source'].shape, (1, 3, 64, 64))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
