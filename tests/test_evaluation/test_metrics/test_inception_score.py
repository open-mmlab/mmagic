# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from mmengine.runner import Runner

from mmagic.datasets import PairedImageDataset
from mmagic.evaluation import InceptionScore, TransIS
from mmagic.models import DataPreprocessor, Pix2Pix
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


def process_fn(data_batch, predictions):

    _predictions = []
    for pred in predictions:
        _predictions.append(pred.to_dict())
    return data_batch, _predictions


class inception_mock(nn.Module):

    def __init__(self, style):
        super().__init__()
        self.style = style

    def forward(self, x, *args, **kwargs):
        mock_feat = torch.randn(x.shape[0], 2048)
        if self.style.upper() in ['STYLEGAN', 'IS']:
            return mock_feat
        else:
            return [mock_feat]


class TestIS(TestCase):

    mock_inception_stylegan = MagicMock(
        return_value=(inception_mock('IS'), 'StyleGAN'))
    mock_inception_pytorch = MagicMock(
        return_value=(inception_mock('IS'), 'PyTorch'))

    def test_init(self):
        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(fake_nums=2, fake_key='fake')

        from PIL import Image
        self.assertEqual(IS.resize, True)
        self.assertEqual(IS.splits, 10)
        self.assertEqual(IS.resize_method, Image.BICUBIC)

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', use_pillow_resize=False)
        self.assertEqual(IS.use_pillow_resize, False)
        self.assertEqual(IS.resize_method, 'bicubic')

        module = MagicMock()
        module.data_preprocessor = MagicMock()
        module.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()
        IS.prepare(module, dataloader)

    # NOTE: do not test load inception network to save time
    # def test_load_inception(self):
    #     IS = InceptionScore(fake_nums=2, inception_style='PyTorch')
    #     self.assertEqual(IS.inception_style.upper(), 'PYTORCH')

    def test_process_and_compute(self):
        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(fake_nums=2, fake_key='fake')
        gen_images = torch.randn(4, 3, 2, 2)
        gen_samples = [
            DataSample(fake_img=img).to_dict() for img in gen_images
        ]
        IS.process(None, gen_samples)
        IS.process(None, gen_samples)

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_pytorch):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', inception_style='PyTorch')
        gen_images = torch.randn(4, 3, 2, 2)
        gen_samples = [
            DataSample(fake_img=img).to_dict() for img in gen_images
        ]
        IS.process(None, gen_samples)

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', sample_model='orig')
        gen_samples = [
            DataSample(
                ema=DataSample(fake_img=torch.randn(3, 2, 2)),
                orig=DataSample(fake_img=torch.randn(3, 2, 2))).to_dict()
        ]
        IS.process(None, gen_samples)
        gen_samples = [
            DataSample(orig=DataSample(fake=torch.randn(3, 2, 2))).to_dict()
        ]
        IS.process(None, gen_samples)

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', sample_model='orig')
        gen_samples = [
            DataSample(fake_img=torch.randn(3, 2, 2)).to_dict()
            for _ in range(4)
        ]
        IS.process(None, gen_samples)

        metric = IS.evaluate()
        self.assertIsInstance(metric, dict)
        self.assertTrue('is' in metric)
        self.assertTrue('is_std' in metric)


class TestTransIS:
    inception_pkl = osp.join(
        osp.dirname(__file__), '..', '..',
        'data/inception_pkl/inception_feat.pkl')

    mock_inception_stylegan = MagicMock(
        return_value=(inception_mock('StyleGAN'), 'StyleGAN'))
    mock_inception_pytorch = MagicMock(
        return_value=(inception_mock('PyTorch'), 'PyTorch'))

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(
                type='LoadPairedImageFromFile',
                key='pair',
                domain_a='edge',
                domain_b='shoe',
                color_type='color'),
            dict(
                type='TransformBroadcaster',
                mapping={'img': ['img_edge', 'img_shoe']},
                auto_remap=True,
                share_random_params=True,
                transforms=[
                    dict(
                        type='Resize',
                        scale=(286, 286),
                        interpolation='bicubic'),
                    dict(type='FixedCrop', keys=['img'], crop_size=(256, 256))
                ]),
            dict(type='PackInputs', keys=['img_edge', 'img_shoe', 'pair'])
        ]
        dataset = PairedImageDataset(
            data_root='tests/data/paired', pipeline=pipeline, test_mode=True)
        cls.dataloader = Runner.build_dataloader(
            dict(
                batch_size=2,
                dataset=dataset,
                sampler=dict(type='DefaultSampler')))
        gan_data_preprocessor = DataPreprocessor()
        generator = dict(
            type='UnetGenerator',
            in_channels=3,
            out_channels=3,
            num_down=8,
            base_channels=64,
            norm_cfg=dict(type='BN'),
            use_dropout=True,
            init_cfg=dict(type='normal', gain=0.02))
        discriminator = dict(
            type='PatchDiscriminator',
            in_channels=6,
            base_channels=64,
            num_conv=3,
            norm_cfg=dict(type='BN'),
            init_cfg=dict(type='normal', gain=0.02))
        cls.module = Pix2Pix(
            generator,
            discriminator,
            data_preprocessor=gan_data_preprocessor,
            default_domain='shoe',
            reachable_domains=['shoe'],
            related_domains=['shoe', 'edge'])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_trans_is_cuda(self):
        return
        with patch.object(TransIS, '_load_inception',
                          self.mock_inception_stylegan):
            IS = TransIS(
                prefix='IS-Full',
                fake_nums=2,
                inception_style='PyTorch',
                fake_key='fake_shoe',
                sample_model='orig')
        self.module.cuda()
        sampler = IS.get_metric_sampler(self.module, self.dataloader, [IS])
        IS.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            IS.process(_data_batch, _predictions)
        IS_res = IS.compute_metrics(IS.fake_results)
        assert 'is' in IS_res and 'is_std' in IS_res

    def test_trans_is_cpu(self):
        return
        with patch.object(TransIS, '_load_inception',
                          self.mock_inception_stylegan):
            IS = TransIS(
                prefix='IS-Full',
                fake_nums=2,
                inception_style='PyTorch',
                fake_key='fake_shoe',
                sample_model='orig')
        sampler = IS.get_metric_sampler(self.module, self.dataloader, [IS])
        IS.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            IS.process(_data_batch, _predictions)
        IS_res = IS.compute_metrics(IS.fake_results)
        assert 'is' in IS_res and 'is_std' in IS_res


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
