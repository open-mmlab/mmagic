# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
from unittest import TestCase

from mmagic.registry import DIFFUSION_SCHEDULERS

test_dir = osp.join(osp.dirname(__file__), '../../..', 'tests')
config_path = osp.join(test_dir, 'configs', 'scheduler_cfg')


class TestWrapper(TestCase):

    def test_build(self):
        # 1. test init by args
        config = dict(
            type='EulerDiscreteScheduler',
            num_train_timesteps=2000,
            beta_schedule='scaled_linear')
        scheduler = DIFFUSION_SCHEDULERS.build(config)
        self.assertEqual(len(scheduler.timesteps), 2000)
        # self.assertEqual(scheduler.beta_schedule, 'scaled_linear')
        scheduler_str = repr(scheduler)
        self.assertIn(
            'Wrapped Scheduler Class: '
            '<class \'diffusers.schedulers.'
            'scheduling_euler_discrete.'
            'EulerDiscreteScheduler\'>', scheduler_str)
        self.assertIn('Wrapped Scheduler Name: EulerDiscreteScheduler',
                      scheduler_str)
        self.assertNotIn('From Pretrained: ', scheduler_str)

        # 2. test save as diffuser
        scheduler.save_pretrained(config_path)

        # 3. test from_pretrained
        config = dict(
            type='EulerDiscreteScheduler', from_pretrained=config_path)
        scheduler = DIFFUSION_SCHEDULERS.build(config)
        scheduler_str = repr(scheduler)
        self.assertIn('From Pretrained: ', scheduler_str)

        # 4. test attribute error
        with self.assertRaises(AttributeError):
            scheduler.unsupported_attr('do not support')

        # tear down
        shutil.rmtree(config_path)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
