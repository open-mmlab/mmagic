# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import pickle
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmengine.runner import Runner

from mmagic.datasets import PairedImageDataset
from mmagic.evaluation import FrechetInceptionDistance, TransFID
from mmagic.models import DataPreprocessor, Pix2Pix
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


def process_fn(data_batch, predictions):

    _predictions = []
    for pred in predictions:
        _predictions.append(pred.to_dict())
    return data_batch, _predictions


def construct_inception_pkl(inception_path):
    data_root = osp.dirname(inception_path)
    os.makedirs(data_root, exist_ok=True)
    with open(inception_path, 'wb') as file:
        feat = np.random.rand(10, 2048)
        mean = np.mean(feat, 0)
        cov = np.cov(feat, rowvar=False)
        inception_feat = dict(raw_feature=feat, real_mean=mean, real_cov=cov)
        pickle.dump(inception_feat, file)


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


class TestFID(TestCase):

    inception_pkl = osp.join(
        osp.dirname(__file__), '..', '..',
        'data/inception_pkl/inception_feat.pkl')

    mock_inception_stylegan = MagicMock(
        return_value=(inception_mock('StyleGAN'), 'StyleGAN'))
    mock_inception_pytorch = MagicMock(
        return_value=(inception_mock('PyTorch'), 'PyTorch'))

    def test_init(self):
        construct_inception_pkl(self.inception_pkl)

        with patch.object(FrechetInceptionDistance, '_load_inception',
                          self.mock_inception_stylegan):

            fid = FrechetInceptionDistance(
                fake_nums=2,
                real_key='real',
                fake_key='fake',
                inception_pkl=self.inception_pkl)

            self.assertFalse(fid.need_cond_input)
            self.assertIsNone(fid.real_mean)
            self.assertIsNone(fid.real_cov)

        module = MagicMock()
        module.data_preprocessor = MagicMock()
        module.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()
        fid.prepare(module, dataloader)

        self.assertIsNotNone(fid.real_mean)
        self.assertIsNotNone(fid.real_cov)

    def test_prepare(self):

        module = MagicMock()
        module.data_preprocessor = MagicMock()
        module.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()

        with patch.object(FrechetInceptionDistance, '_load_inception',
                          self.mock_inception_stylegan):
            fid = FrechetInceptionDistance(
                fake_nums=2,
                real_nums=2,
                real_key='real',
                fake_key='fake',
                inception_pkl=self.inception_pkl)
        fid.prepare(module, dataloader)

    # NOTE: do not test load inception network to save time
    # def test_load_inception(self):
    #     fid = FrechetInceptionDistance(
    #         fake_nums=2,
    #         real_nums=2,
    #         real_key='real',
    #         fake_key='fake',
    #         inception_style='PyTorch',
    #         inception_pkl=self.inception_pkl)
    #     self.assertEqual(fid.inception_style.upper(), 'PYTORCH')

    def test_process_and_compute(self):
        with patch.object(FrechetInceptionDistance, '_load_inception',
                          self.mock_inception_stylegan):
            fid = FrechetInceptionDistance(
                fake_nums=2,
                real_nums=2,
                real_key='real',
                fake_key='fake',
                inception_pkl=self.inception_pkl)
        gen_images = torch.randn(4, 3, 2, 2)
        gen_samples = [
            DataSample(fake=(gen_images[i])).to_dict() for i in range(4)
        ]
        fid.process(None, gen_samples)
        fid.process(None, gen_samples)

        fid.fake_results.clear()
        gen_sample = [
            DataSample(orig=DataSample(fake=torch.randn(3, 2, 2))).to_dict()
        ]
        fid.process(None, gen_sample)
        gen_sample = [
            DataSample(orig=DataSample(
                fake_img=torch.randn(3, 2, 2))).to_dict()
        ]
        fid.process(None, gen_sample)

        with patch.object(FrechetInceptionDistance, '_load_inception',
                          self.mock_inception_pytorch):
            fid = FrechetInceptionDistance(
                fake_nums=2,
                real_nums=2,
                real_key='real',
                fake_key='fake',
                inception_style='PyTorch',
                inception_pkl=self.inception_pkl)
        module = MagicMock()
        module.data_preprocessor = MagicMock()
        module.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()
        fid.prepare(module, dataloader)
        gen_samples = [
            DataSample(fake_img=torch.randn(3, 2, 2)).to_dict()
            for _ in range(4)
        ]
        fid.process(None, gen_samples)

        metric = fid.evaluate()
        self.assertIsInstance(metric, dict)
        self.assertTrue('fid' in metric)
        self.assertTrue('mean' in metric)
        self.assertTrue('cov' in metric)


class TestTransFID:
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
                        scale=(256, 256),
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
    def test_trans_fid_cuda(self):
        return
        with patch.object(TransFID, '_load_inception',
                          self.mock_inception_stylegan):
            fid = TransFID(
                prefix='FID-Full',
                fake_nums=2,
                real_key='img_shoe',
                fake_key='fake_shoe',
                inception_style='PyTorch')
        self.module.cuda()
        sampler = fid.get_metric_sampler(self.module, self.dataloader, [fid])
        fid.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            fid.process(_data_batch, _predictions)
        fid_res = fid.compute_metrics(fid.fake_results)
        assert fid_res['fid'] >= 0 and fid_res['mean'] >= 0 and fid_res[
            'cov'] >= 0

    def test_trans_fid_cpu(self):
        return
        with patch.object(TransFID, '_load_inception',
                          self.mock_inception_stylegan):
            fid = TransFID(
                prefix='FID-Full',
                fake_nums=2,
                real_key='img_shoe',
                fake_key='fake_shoe',
                inception_style='PyTorch')
        sampler = fid.get_metric_sampler(self.module, self.dataloader, [fid])
        fid.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            fid.process(_data_batch, _predictions)
        fid_res = fid.compute_metrics(fid.fake_results)
        assert fid_res['fid'] >= 0 and fid_res['mean'] >= 0 and fid_res[
            'cov'] >= 0


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
