# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase
from unittest.mock import MagicMock

from mmengine.model import MMSeparateDistributedDataParallel

from mmagic.engine import PGGANOptimWrapperConstructor
from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()


class TestPGGANOptimWrapperConstructor(TestCase):

    pggan_cfg = dict(
        type='ProgressiveGrowingGAN',
        data_preprocessor=dict(type='DataPreprocessor'),
        noise_size=512,
        generator=dict(type='PGGANGenerator', out_scale=8),
        discriminator=dict(type='PGGANDiscriminator', in_scale=8),
        nkimgs_per_scale={
            '4': 600,
            '8': 1200
        },
        transition_kimgs=600,
        ema_config=dict(interval=1))

    base_lr = 0.001
    lr_schedule = dict(generator={'8': 0.0015}, discriminator={'8': 0.0015})
    optim_wrapper_cfg = dict(
        generator=dict(
            optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
        discriminator=dict(
            optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
        lr_schedule=lr_schedule)

    def test(self):
        pggan = MODELS.build(self.pggan_cfg)
        optim_wrapper_dict_builder = PGGANOptimWrapperConstructor(
            self.optim_wrapper_cfg)
        optim_wrapper_dict = optim_wrapper_dict_builder(pggan)
        optim_keys = set(optim_wrapper_dict.keys())

        scales = self.pggan_cfg['nkimgs_per_scale'].keys()
        self.assertEqual(
            optim_keys,
            set([
                f'{model}_{scale}' for model in ['generator', 'discriminator']
                for scale in scales
            ]))

        # check lr
        for scale in scales:
            gen_optim = optim_wrapper_dict[f'generator_{scale}']
            gen_lr = gen_optim.optimizer.param_groups[0]['lr']

            disc_optim = optim_wrapper_dict[f'discriminator_{scale}']
            disc_lr = disc_optim.optimizer.param_groups[0]['lr']

            self.assertEqual(
                gen_lr,
                self.lr_schedule['generator'].get(str(scale), self.base_lr))
            self.assertEqual(
                disc_lr, self.lr_schedule['discriminator'].get(
                    str(scale), self.base_lr))

        # test pggan is Wrapper
        pggan_with_wrapper = MagicMock(
            module=pggan, spec=MMSeparateDistributedDataParallel)
        optim_wrapper_dict = optim_wrapper_dict_builder(pggan_with_wrapper)

        # test raise error
        with self.assertRaises(TypeError):
            optim_wrapper_dict_builder = PGGANOptimWrapperConstructor(
                'optim_wrapper')

        # test same optimizer
        optim_wrapper_cfg = deepcopy(self.optim_wrapper_cfg)
        optim_wrapper_cfg['reset_optim_for_new_scale'] = False
        optim_wrapper_dict_builder = PGGANOptimWrapperConstructor(
            optim_wrapper_cfg)
        optim_wrapper_dict = optim_wrapper_dict_builder(pggan)
        # check id are same
        gen_optims = [optim_wrapper_dict[k] for k in optim_keys if 'gen' in k]
        disc_optims = [
            optim_wrapper_dict[k] for k in optim_keys if 'disc' in k
        ]
        self.assertTrue(all([id(gen_optims[0]) == id(o) for o in gen_optims]))
        self.assertTrue(
            all([id(disc_optims[0]) == id(o) for o in disc_optims]))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
