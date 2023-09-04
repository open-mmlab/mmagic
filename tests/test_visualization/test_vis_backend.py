# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import sys
import time
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch
from mmengine import Config, MessageHub

from mmagic.visualization import (PaviVisBackend, TensorboardVisBackend,
                                  VisBackend, WandbVisBackend)


class TestVisBackend(TestCase):

    def test_vis_backend(self):
        message_hub = MessageHub.get_instance('test-visbackend')
        config = Config(dict(work_dir='./mmagic/test/vis_backend_test/'))
        message_hub.update_info('cfg', config.pretty_text)

        data_root = 'tmp_dir'
        sys.modules['petrel_client'] = MagicMock()
        vis_backend = VisBackend(save_dir='tmp_dir', ceph_path='s3://xxx')

        self.assertEqual(vis_backend.experiment, vis_backend)
        # test path mapping
        src_path = osp.abspath(
            './mmagic/test/vis_backend_test/test_vis_data/test.png')
        tar_path = 's3://xxx/vis_backend_test/test_vis_data/test.png'
        file_client = vis_backend._file_client
        mapped_path = file_client._map_path(src_path)
        formatted_path = file_client._format_path(mapped_path)
        self.assertEqual(formatted_path, tar_path)
        # test with `delete_local` is True
        vis_backend.add_config(Config(dict(name='test')))
        vis_backend.add_image(
            name='add_img', image=np.random.random((8, 8, 3)).astype(np.uint8))
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3)

        # test with `delete_local` is False
        vis_backend._delete_local_image = False
        vis_backend.add_config(Config(dict(name='test')))
        vis_backend.add_image(
            name='add_img', image=np.random.random((8, 8, 3)).astype(np.uint8))
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3)
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3,
            file_path='new_scalar.json')
        self.assertTrue(osp.exists(osp.join(data_root, 'config.py')))
        self.assertTrue(
            osp.exists(osp.join(data_root, 'vis_image', 'add_img_0.png')))
        self.assertTrue(osp.exists(osp.join(data_root, 'new_scalar.json')))
        self.assertTrue(osp.exists(osp.join(data_root, 'scalars.json')))

        # test with `ceph_path` is None
        vis_backend = VisBackend(save_dir='tmp_dir')
        vis_backend.add_config(Config(dict(name='test')))
        vis_backend.add_image(
            name='add_img', image=np.random.random((8, 8, 3)).astype(np.uint8))
        vis_backend.add_scalar(
            name='scalar_tensor', value=torch.FloatTensor([0.693]), step=3)
        vis_backend.add_scalar(name='scalar', value=0.693, step=3)
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3)
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3,
            file_path='new_scalar.json')

        # raise error
        with self.assertRaises(AssertionError):
            vis_backend.add_scalars(
                scalar_dict=dict(lr=0.001), step=3, file_path='scalars.json')
        with self.assertRaises(AssertionError):
            vis_backend.add_scalars(
                scalar_dict=dict(lr=0.001), step=3, file_path='new_scalars')

        shutil.rmtree('tmp_dir')


class TestTensorboardBackend(TestCase):

    def test_tensorboard(self):
        save_dir = 'tmp_dir'
        vis_backend = TensorboardVisBackend(save_dir)
        sys.modules['torch.utils.tensorboard'] = MagicMock()

        # add image
        vis_backend.add_image(
            name='add_img', image=np.random.random((8, 8, 3)).astype(np.uint8))
        vis_backend.add_image(
            name='add_img',
            image=np.random.random((10, 8, 8, 3)).astype(np.uint8))

        # add scalars
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3)


class TestPaviBackend(TestCase):

    def test_pavi(self):
        save_dir = 'tmp_dir'
        exp_name = 'unit test'
        vis_backend = PaviVisBackend(save_dir=save_dir, exp_name=exp_name)
        with self.assertRaises(ImportError):
            vis_backend._init_env()
        sys.modules['pavi'] = MagicMock()
        vis_backend._init_env()

        exp = vis_backend.experiment
        self.assertEqual(exp, vis_backend._pavi)

        # add image
        vis_backend.add_image(
            name='add_img', image=np.random.random((8, 8, 3)).astype(np.uint8))

        # add scalars
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3)


class TestWandbBackend(TestCase):

    def test_wandb(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

        wandb_mock = MagicMock()
        sys.modules['wandb'] = wandb_mock

        vis_backend = WandbVisBackend(
            save_dir=f'parent_dir/exp_name/{timestamp}/vis_data',
            init_kwargs=dict(project='test_backend'))
        # test save gif image
        rgb_np = np.random.rand(11, 4, 4, 3).astype(np.uint8)
        vis_backend.add_image('test_gif', rgb_np, n_skip=2)
        vis_backend.add_image('test_gif', rgb_np, n_skip=1)

        # test save rgb image
        rgb_np = np.random.rand(4, 4, 3).astype(np.uint8)
        vis_backend.add_image('test_rgb', rgb_np)

        # test wandb backend with name
        wandb_mock.reset_mock()
        vis_backend = WandbVisBackend(
            save_dir=f'parent_dir/exp_name/{timestamp}/vis_data',
            init_kwargs=dict(project='test_backend', name='test_wandb'))
        vis_backend._init_env()
        _, called_kwargs = wandb_mock.init.call_args
        self.assertEqual(called_kwargs['name'], f'test_wandb_{timestamp}')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
