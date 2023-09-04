# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase
from unittest.mock import MagicMock, Mock

import mmcv
import numpy as np
import torch
from mmengine import MessageHub
from mmengine.testing import assert_allclose
from mmengine.visualization import Visualizer
from torch.utils.data.dataset import Dataset

from mmagic.engine import VisualizationHook
from mmagic.engine.hooks import BasicVisualizationHook
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules
from mmagic.visualization import ConcatImageVisualizer

from mmagic.registry import MODELS  # isort:skip  # noqa

register_all_modules()


class TestBasicVisualizationHook(TestCase):

    def setUp(self) -> None:
        input = torch.rand(2, 3, 32, 32)
        data_sample = DataSample(
            path_rgb='rgb.png',
            tensor3d=torch.ones(3, 32, 32) *
            torch.tensor([[[0.1]], [[0.2]], [[0.3]]]),
            array3d=np.ones(shape=(32, 32, 3)) * [0.4, 0.5, 0.6],
            tensor4d=torch.ones(2, 3, 32, 32) * torch.tensor(
                [[[[0.1]], [[0.2]], [[0.3]]], [[[0.4]], [[0.5]], [[0.6]]]]),
            pixdata=torch.ones(1, 32, 32) * 0.6)
        self.data_batch = {'inputs': input, 'data_samples': [data_sample] * 2}

        output = copy.deepcopy(data_sample)
        output.outpixdata = np.ones(shape=(32, 32)) * 0.8
        self.outputs = [output] * 2

        self.vis = ConcatImageVisualizer(
            fn_key='path_rgb',
            img_keys=[
                'tensor3d',
                'array3d',
                'pixdata',
                'tensor4d',
                'outpixdata',
            ],
            vis_backends=[dict(type='LocalVisBackend')],
            save_dir='work_dirs')

    def test_after_iter(self):
        runner = Mock()
        runner.iter = 1
        runner.visualizer = self.vis
        hook = BasicVisualizationHook()
        hook._after_iter(runner, 1, self.data_batch, self.outputs)

        img = mmcv.imread('work_dirs/vis_data/vis_image/rgb_1.png')
        assert img.shape == (64, 160, 3)

        hook._on_train = hook._on_test = hook._on_val = True
        runner.visualizer = Mock()
        runner.visualizer.add_datasample = Mock()
        hook._after_iter(
            runner, 1, self.data_batch, self.outputs, mode='train')
        runner.visualizer.add_datasample.assert_called
        hook._after_iter(runner, 1, self.data_batch, self.outputs, mode='val')
        runner.visualizer.add_datasample.assert_called
        hook._after_iter(runner, 1, self.data_batch, self.outputs, mode='test')
        assert runner.visualizer.assert_called

        hook._on_train = hook._on_test = hook._on_val = False
        hook._after_iter(
            runner, 1, self.data_batch, self.outputs, mode='train')
        assert runner.visualizer.assert_not_called
        hook._after_iter(runner, 1, self.data_batch, self.outputs, mode='val')
        assert runner.visualizer.assert_not_called
        hook._after_iter(runner, 1, self.data_batch, self.outputs, mode='test')
        assert runner.visualizer.assert_not_called


class TestVisualizationHook(TestCase):

    Visualizer.get_instance('test-gen-visualizer')
    MessageHub.get_instance('test-gen-visualizer')

    def test_init(self):
        hook = VisualizationHook(
            interval=10, vis_kwargs_list=dict(type='Noise'))
        self.assertEqual(hook.interval, 10)
        self.assertEqual(hook.vis_kwargs_list, [dict(type='Noise')])
        self.assertEqual(hook.n_samples, 64)
        self.assertFalse(hook.show)

        hook = VisualizationHook(
            interval=10,
            vis_kwargs_list=[dict(type='Noise'),
                             dict(type='Translation')])
        self.assertEqual(len(hook.vis_kwargs_list), 2)

        hook = VisualizationHook(
            interval=10, vis_kwargs_list=dict(type='GAN'), show=True)
        self.assertEqual(hook._visualizer._vis_backends, {})

    def test_vis_sample_with_gan_alias(self):
        gan_model_cfg = dict(
            type='DCGAN',
            noise_size=10,
            data_preprocessor=dict(type='DataPreprocessor'),
            generator=dict(
                type='DCGANGenerator', output_scale=32, base_channels=32))
        model = MODELS.build(gan_model_cfg)
        runner = MagicMock()
        runner.model = model
        runner.train_dataloader = MagicMock()
        runner.train_dataloader.batch_size = 10

        hook = VisualizationHook(
            interval=10, vis_kwargs_list=dict(type='GAN'), n_samples=9)
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        # build a empty data sample
        data_batch = [
            dict(inputs=None, data_samples=DataSample()) for idx in range(10)
        ]
        hook.vis_sample(runner, 0, data_batch, None)
        _, called_kwargs = mock_visualuzer.add_datasample.call_args
        self.assertEqual(called_kwargs['name'], 'gan')
        self.assertEqual(called_kwargs['target_keys'], None)
        self.assertEqual(called_kwargs['vis_mode'], None)
        gen_batches = called_kwargs['gen_samples']
        self.assertEqual(len(gen_batches), 9)
        noise_in_gen_batches = torch.stack(
            [gen_batches[idx].noise for idx in range(9)], 0)
        noise_in_buffer = torch.cat([
            buffer['inputs']['noise'] for buffer in hook.inputs_buffer['GAN']
        ],
                                    dim=0)[:9]
        self.assertTrue((noise_in_gen_batches == noise_in_buffer).all())

        hook.vis_sample(runner, 1, data_batch, None)
        _, called_kwargs = mock_visualuzer.add_datasample.call_args
        gen_batches = called_kwargs['gen_samples']
        noise_in_gen_batches_new = torch.stack(
            [gen_batches[idx].noise for idx in range(9)], 0)
        self.assertTrue((noise_in_gen_batches_new == noise_in_buffer).all())

    def test_vis_sample_with_translation_alias(self):
        translation_cfg = dict(
            type='CycleGAN',
            data_preprocessor=dict(
                type='DataPreprocessor', data_keys=['img_photo', 'img_mask']),
            generator=dict(
                type='ResnetGenerator',
                in_channels=3,
                out_channels=3,
                base_channels=8,
                norm_cfg=dict(type='IN'),
                use_dropout=False,
                num_blocks=4,
                padding_mode='reflect',
                init_cfg=dict(type='normal', gain=0.02)),
            discriminator=dict(
                type='PatchDiscriminator',
                in_channels=3,
                base_channels=8,
                num_conv=3,
                norm_cfg=dict(type='IN'),
                init_cfg=dict(type='normal', gain=0.02)),
            default_domain='photo',
            reachable_domains=['photo', 'mask'],
            related_domains=['photo', 'mask'])
        model = MODELS.build(translation_cfg)

        class naive_dataset(Dataset):

            def __init__(self, max_len, train=False):
                self.max_len = max_len
                self.train = train

            def __len__(self):
                return self.max_len

            def __getitem__(self, index):
                weight = index if self.train else -index
                img_photo = torch.ones(3, 32, 32) * weight
                img_mask = torch.ones(3, 32, 32) * (weight + 1)
                return dict(
                    inputs=dict(img_photo=img_photo, img_mask=img_mask),
                    data_samples=DataSample(
                        img_photo=img_photo, img_mask=img_mask))

        train_dataloader = MagicMock()
        train_dataloader.batch_size = 4
        train_dataloader.dataset = naive_dataset(max_len=15, train=True)
        val_dataloader = MagicMock()
        val_dataloader.batch_size = 4
        val_dataloader.dataset = naive_dataset(max_len=17)

        runner = MagicMock()
        runner.model = model
        runner.train_dataloader = train_dataloader
        runner.val_loop = MagicMock()
        runner.val_loop.dataloader = val_dataloader

        hook = VisualizationHook(
            interval=10,
            vis_kwargs_list=[
                dict(type='Translation'),
                dict(type='TranslationVal', name='cyclegan_val')
            ],
            n_samples=9)
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        # build a empty data sample
        data_batch = [
            dict(inputs=None, data_samples=DataSample()) for idx in range(4)
        ]
        hook.vis_sample(runner, 0, data_batch, None)
        called_kwargs_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(len(called_kwargs_list), 2)
        # trans_called_kwargs, trans_val_called_kwargs = called_kwargs_list
        _, trans_called_kwargs = called_kwargs_list[0]
        _, trans_val_called_kwargs = called_kwargs_list[1]
        self.assertEqual(trans_called_kwargs['name'], 'translation')
        self.assertEqual(trans_val_called_kwargs['name'], 'cyclegan_val')

        # test train gen samples
        trans_gen_sample = trans_called_kwargs['gen_samples']
        print(trans_gen_sample[0].keys())
        trans_gt_mask_list = [samp.img_mask for samp in trans_gen_sample]
        trans_gt_photo_list = [samp.img_photo for samp in trans_gen_sample]

        self.assertEqual(len(trans_gen_sample), 9)
        for idx, (mask, photo) in enumerate(
                zip(trans_gt_mask_list, trans_gt_photo_list)):
            sample_from_dataset = train_dataloader.dataset[idx]['inputs']
            # data sample in test mode --> do not normed
            assert_allclose(mask, sample_from_dataset['img_mask'])
            assert_allclose(photo, sample_from_dataset['img_photo'])

        # test val gen samples
        trans_gen_sample = trans_val_called_kwargs['gen_samples']
        trans_gt_mask_list = [samp.img_mask for samp in trans_gen_sample]
        trans_gt_photo_list = [samp.img_photo for samp in trans_gen_sample]

        self.assertEqual(len(trans_gen_sample), 9)
        for idx, (mask, photo) in enumerate(
                zip(trans_gt_mask_list, trans_gt_photo_list)):
            sample_from_dataset = val_dataloader.dataset[idx]['inputs']
            assert_allclose(mask, sample_from_dataset['img_mask'])
            assert_allclose(photo, sample_from_dataset['img_photo'])

        # check input buffer
        input_buffer = hook.inputs_buffer
        input_buffer['translation']

    # TODO: uncomment after support DDPM
    # def test_vis_ddpm_alias_with_user_defined_args(self):
    #     ddpm_cfg = dict(
    #         type='BasicGaussianDiffusion',
    #         num_timesteps=4,
    #         data_preprocessor=dict(type='DataPreprocessor'),
    #         betas_cfg=dict(type='cosine'),
    #         denoising=dict(
    #             type='DenoisingUnet',
    #             image_size=32,
    #             in_channels=3,
    #             base_channels=128,
    #             resblocks_per_downsample=3,
    #             attention_res=[16, 8],
    #             use_scale_shift_norm=True,
    #             dropout=0.3,
    #             num_heads=4,
    #             use_rescale_timesteps=True,
    #             output_cfg=dict(mean='eps', var='learned_range')),
    #         timestep_sampler=dict(type='UniformTimeStepSampler'))
    #     model = MODELS.build(ddpm_cfg)
    #     runner = MagicMock()
    #     runner.model = model
    #     runner.train_dataloader = MagicMock()
    #     runner.train_dataloader.batch_size = 10

    #     hook = VisualizationHook(
    #         interval=10,
    #         n_samples=2,
    #         vis_kwargs_list=dict(
    #             type='DDPMDenoising', vis_mode='gif', name='ddpm',
    #             n_samples=3))
    #     mock_visualuzer = MagicMock()
    #     mock_visualuzer.add_datasample = MagicMock()
    #     hook._visualizer = mock_visualuzer

    #     # build a empty data sample
    #     data_batch = [
    #         dict(inputs=None, data_sample=DataSample())
    #         for idx in range(10)
    #     ]
    #     hook.vis_sample(runner, 0, data_batch, None)
    #     _, called_kwargs = mock_visualuzer.add_datasample.call_args
    #     gen_samples = called_kwargs['gen_samples']
    #     self.assertEqual(len(gen_samples), 3)
    #     self.assertEqual(called_kwargs['n_row'], min(hook.n_row, 3))

    #     # test user defined vis kwargs
    #     hook.vis_kwargs_list = [
    #         dict(
    #             type='Arguments',
    #             forward_mode='sampling',
    #             name='ddpm_sample',
    #             n_samples=2,
    #             n_row=4,
    #             vis_mode='gif',
    #             n_skip=1,
    #             forward_kwargs=dict(
    #                 forward_mode='sampling',
    #                 sample_kwargs=dict(show_pbar=True, save_intermedia=True)))  # noqa
    #     ]
    #     mock_visualuzer = MagicMock()
    #     mock_visualuzer.add_datasample = MagicMock()
    #     hook._visualizer = mock_visualuzer

    #     # build a empty data sample
    #     data_batch = [
    #         dict(inputs=None, data_sample=DataSample())
    #         for idx in range(10)
    #     ]
    #     hook.vis_sample(runner, 0, data_batch, None)
    #     _, called_kwargs = mock_visualuzer.add_datasample.call_args
    #     gen_samples = called_kwargs['gen_samples']
    #     self.assertEqual(len(gen_samples), 2)
    #     self.assertEqual(called_kwargs['n_row'], min(hook.n_row, 2))

    def test_after_val_iter(self):
        model = MagicMock()
        hook = VisualizationHook(
            interval=10, n_samples=2, vis_kwargs_list=dict(type='GAN'))
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        runner = MagicMock()
        runner.model = model

        hook.after_val_iter(runner, 0, [dict()], [DataSample()])
        mock_visualuzer.assert_not_called()

    def test_after_train_iter(self):
        gan_model_cfg = dict(
            type='DCGAN',
            noise_size=10,
            data_preprocessor=dict(type='DataPreprocessor'),
            generator=dict(
                type='DCGANGenerator', output_scale=32, base_channels=32))
        model = MODELS.build(gan_model_cfg)
        runner = MagicMock()
        runner.model = model
        runner.train_dataloader = MagicMock()
        runner.train_dataloader.batch_size = 10

        hook = VisualizationHook(
            interval=2, vis_kwargs_list=dict(type='GAN'), n_samples=9)
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        # build a empty data sample
        data_batch = [
            dict(inputs=None, data_samples=DataSample()) for idx in range(10)
        ]
        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        self.assertEqual(mock_visualuzer.add_datasample.call_count, 1)

        # test vis with messagehub info --> str
        mock_visualuzer.add_datasample.reset_mock()
        message_hub = MessageHub.get_current_instance()

        feat_map = torch.randn(4, 16, 4, 4)
        vis_results = dict(feat_map=feat_map)
        message_hub.update_info('vis_results', vis_results)

        hook.message_vis_kwargs = 'feat_map'
        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        called_args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(len(called_args_list), 2)  # outputs + messageHub
        _, messageHub_vis_args = called_args_list[1]
        self.assertEqual(messageHub_vis_args['name'], 'train_feat_map')
        self.assertEqual(len(messageHub_vis_args['gen_samples']), 4)
        self.assertEqual(messageHub_vis_args['vis_mode'], None)
        self.assertEqual(messageHub_vis_args['n_row'], None)

        # test vis with messagehub info --> list[str]
        mock_visualuzer.add_datasample.reset_mock()

        hook.message_vis_kwargs = ['feat_map']
        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        called_args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(len(called_args_list), 2)  # outputs + messageHub
        _, messageHub_vis_args = called_args_list[1]
        self.assertEqual(messageHub_vis_args['name'], 'train_feat_map')
        self.assertEqual(len(messageHub_vis_args['gen_samples']), 4)
        self.assertEqual(messageHub_vis_args['vis_mode'], None)
        self.assertEqual(messageHub_vis_args['n_row'], None)

        # test vis with messagehub info --> dict
        mock_visualuzer.add_datasample.reset_mock()

        hook.message_vis_kwargs = dict(key='feat_map', vis_mode='feature_map')
        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        called_args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(len(called_args_list), 2)  # outputs + messageHub
        _, messageHub_vis_args = called_args_list[1]
        self.assertEqual(messageHub_vis_args['name'], 'train_feat_map')
        self.assertEqual(len(messageHub_vis_args['gen_samples']), 4)
        self.assertEqual(messageHub_vis_args['vis_mode'], 'feature_map')
        self.assertEqual(messageHub_vis_args['n_row'], None)

        # test vis with messagehub info --> list[dict]
        mock_visualuzer.add_datasample.reset_mock()

        feat_map = torch.randn(4, 16, 4, 4)
        x_t = [DataSample(info='x_t')]
        vis_results = dict(feat_map=feat_map, x_t=x_t)
        message_hub.update_info('vis_results', vis_results)

        hook.message_vis_kwargs = [
            dict(key='feat_map', vis_mode='feature_map'),
            dict(key='x_t')
        ]
        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        called_args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(len(called_args_list), 3)  # outputs + messageHub
        # output_vis_args = called_args_list[0].kwargs
        _, feat_map_vis_args = called_args_list[1]
        self.assertEqual(feat_map_vis_args['name'], 'train_feat_map')
        self.assertEqual(len(feat_map_vis_args['gen_samples']), 4)
        self.assertEqual(feat_map_vis_args['vis_mode'], 'feature_map')
        self.assertEqual(feat_map_vis_args['n_row'], None)

        _, x_t_vis_args = called_args_list[2]
        self.assertEqual(x_t_vis_args['name'], 'train_x_t')
        self.assertEqual(len(x_t_vis_args['gen_samples']), 1)
        self.assertEqual(x_t_vis_args['vis_mode'], None)
        self.assertEqual(x_t_vis_args['n_row'], None)

        # test vis messageHub info --> errors
        hook.message_vis_kwargs = 'error'
        with self.assertRaises(RuntimeError):
            hook.after_train_iter(runner, 1, data_batch, None)

        message_hub.runtime_info.clear()
        with self.assertRaises(RuntimeError):
            hook.after_train_iter(runner, 1, data_batch, None)

        hook.message_vis_kwargs = dict(key='feat_map', vis_mode='feature_map')
        message_hub.update_info('vis_results', dict(feat_map='feat_map'))
        with self.assertRaises(TypeError):
            hook.after_train_iter(runner, 1, data_batch, None)

    def test_after_train_iter_contain_mul_elements(self):
        # test contain_mul_elements + n_row != None
        # n_row = 8, n_samples = 3, batch_size = 2, model_n_samples = 4
        # run math.ceil(3 / 2) = 2 times, visualize 2 * 4 = 8 samples,
        class MockModel:

            def __init__(self, n_samples):
                self.n_samples = n_samples

            def noise_fn(self, *args, **kwargs):
                return torch.randn(2, 2)

            def val_step(self, *args, **kwargs):
                return [DataSample() for _ in range(self.n_samples)]

            def eval(self):
                return self

            def train(self):
                return self

        runner = MagicMock()
        runner.model = MockModel(n_samples=4)
        runner.train_dataloader = MagicMock()
        runner.train_dataloader.batch_size = 2

        hook = VisualizationHook(
            interval=2, vis_kwargs_list=dict(type='GAN'), n_samples=3, n_row=8)
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        # build a empty data sample
        data_batch = [
            dict(inputs=None, data_samples=DataSample()) for idx in range(10)
        ]

        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        self.assertEqual(mock_visualuzer.add_datasample.call_count, 1)
        called_args_list = mock_visualuzer.add_datasample.call_args_list[0]
        self.assertEqual(len(called_args_list[1]['gen_samples']), 8)

    def test_after_test_iter(self):
        model = MagicMock()
        hook = VisualizationHook(
            interval=10,
            n_samples=2,
            max_save_at_test=None,
            test_vis_keys=['ema', 'orig', 'new_model.x_t', 'gt_img'],
            vis_kwargs_list=dict(type='GAN'))
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        runner = MagicMock()
        runner.model = model

        gt_list = [torch.randn(3, 6, 6) for _ in range(4)]
        ema_list = [torch.randn(3, 6, 6) for _ in range(4)]
        orig_list = [torch.randn(3, 6, 6) for _ in range(4)]
        x_t_list = [torch.randn(3, 6, 6) for _ in range(4)]

        outputs = []
        for gt, ema, orig, x_t in zip(gt_list, ema_list, orig_list, x_t_list):
            gen_sample = DataSample(
                gt_img=gt,
                ema=DataSample(fake_img=ema),
                orig=DataSample(fake_img=orig),
                new_model=DataSample(x_t=x_t))
            outputs.append(gen_sample)

        hook.after_test_iter(runner, 42, [], outputs)
        args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(
            len(args_list),
            len(hook.test_vis_keys_list) * len(gt_list))
        # check target consistency
        for idx, args in enumerate(args_list):
            _, called_kwargs = args
            gen_samples = called_kwargs['gen_samples']
            name = called_kwargs['name']
            batch_idx = called_kwargs['step']
            target_keys = called_kwargs['target_keys']

            self.assertEqual(len(gen_samples), 1)
            idx_in_outputs = idx // 4
            self.assertEqual(batch_idx, idx_in_outputs + 42 * len(outputs))
            self.assertEqual(outputs[idx_in_outputs], gen_samples[0])

            # check ema
            if idx % 4 == 0:
                self.assertEqual(target_keys, 'ema')
                self.assertEqual(name, 'test_ema')
            # check orig
            elif idx % 4 == 1:
                self.assertEqual(target_keys, 'orig')
                self.assertEqual(name, 'test_orig')
            # check x_t
            elif idx % 4 == 2:
                self.assertEqual(target_keys, 'new_model.x_t')
                self.assertEqual(name, 'test_new_model_x_t')
            # check gt
            else:
                self.assertEqual(target_keys, 'gt_img')
                self.assertEqual(name, 'test_gt_img')

        # test get target key automatically
        hook.test_vis_keys_list = None
        mock_visualuzer.add_datasample.reset_mock()
        hook.after_test_iter(runner, 42, [], outputs)

        kwargs_list = [
            args[1] for args in mock_visualuzer.add_datasample.call_args_list
        ]
        self.assertTrue(all([kwargs['target_keys'] for kwargs in kwargs_list]))

        # test get target key automatically with error
        outputs = [DataSample(ema=DataSample(fake_img=torch.randn(3, 6, 6)))]
        with self.assertRaises(AssertionError):
            hook.after_test_iter(runner, 42, [], outputs)

        # test max save time
        hook = VisualizationHook(
            interval=10,
            n_samples=2,
            test_vis_keys='ema',
            vis_kwargs_list=dict(type='GAN'),
            max_save_at_test=3)

        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        runner = MagicMock()
        runner.model = model

        ema_list = [torch.randn(3, 6, 6) for _ in range(4)]
        outputs = [
            DataSample(ema=DataSample(fake_img=ema)) for ema in ema_list
        ]
        hook.after_test_iter(runner, 42, [], outputs)
        mock_visualuzer.add_datasample.assert_not_called()

        hook.after_test_iter(runner, 0, [], outputs)
        assert mock_visualuzer.add_datasample.call_count == 3


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
