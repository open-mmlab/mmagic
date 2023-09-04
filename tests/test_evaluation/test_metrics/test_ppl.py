# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest.mock import patch

import pytest
import torch
from mmengine.runner import Runner

from mmagic.datasets import BasicImageDataset
from mmagic.datasets.transforms import PackInputs
from mmagic.evaluation import PerceptualPathLength
from mmagic.models import LSGAN, DataPreprocessor
from mmagic.models.editors.stylegan2 import StyleGAN2Generator
from mmagic.utils import register_all_modules

register_all_modules()


def process_fn(data_batch, predictions):

    _predictions = []
    for pred in predictions:
        _predictions.append(pred.to_dict())
    return data_batch, _predictions


class LPIPS_mock:

    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self

    def __call__(self, x1, x2, *args, **kwargs):
        num_batche = x1.shape[0]
        return torch.rand(num_batche, 1, 1, 1)


class TestPPL:

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(type='LoadImageFromFile', key='img'),
            dict(type='Resize', scale=(64, 64)),
            PackInputs()
        ]
        dataset = BasicImageDataset(
            data_root='tests/data/image/img_root',
            pipeline=pipeline,
            test_mode=True)
        cls.dataloader = Runner.build_dataloader(
            dict(
                batch_size=2,
                dataset=dataset,
                sampler=dict(type='DefaultSampler')))
        gan_data_preprocessor = DataPreprocessor()
        generator = StyleGAN2Generator(64, 8)
        cls.module = LSGAN(generator, data_preprocessor=gan_data_preprocessor)

    @patch('lpips.LPIPS', LPIPS_mock)
    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_ppl_cuda(self):
        ppl = PerceptualPathLength(
            fake_nums=2,
            prefix='ppl-z',
            space='Z',
            sample_model='orig',
            latent_dim=8)
        self.module.cuda()
        sampler = ppl.get_metric_sampler(self.module, self.dataloader, [ppl])
        ppl.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            ppl.process(_data_batch, _predictions)
        ppl_res = ppl.compute_metrics(ppl.fake_results)
        assert ppl_res['ppl_score'] >= 0
        ppl = PerceptualPathLength(
            fake_nums=2,
            prefix='ppl-w',
            space='W',
            sample_model='orig',
            latent_dim=8)
        sampler = ppl.get_metric_sampler(self.module, self.dataloader, [ppl])
        ppl.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            ppl.process(_data_batch, _predictions)
        ppl_res = ppl.compute_metrics(ppl.fake_results)
        assert ppl_res['ppl_score'] >= 0

    @patch('lpips.LPIPS', LPIPS_mock)
    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_ppl_cpu(self):
        ppl = PerceptualPathLength(
            fake_nums=2,
            prefix='ppl-z',
            space='Z',
            sample_model='orig',
            latent_dim=8)
        sampler = ppl.get_metric_sampler(self.module, self.dataloader, [ppl])
        ppl.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            ppl.process(_data_batch, _predictions)
        ppl_res = ppl.compute_metrics(ppl.fake_results)
        assert ppl_res['ppl_score'] >= 0
        ppl = PerceptualPathLength(
            fake_nums=2,
            prefix='ppl-w',
            space='W',
            sample_model='orig',
            latent_dim=8)
        sampler = ppl.get_metric_sampler(self.module, self.dataloader, [ppl])
        ppl.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            ppl.process(_data_batch, _predictions)
        ppl_res = ppl.compute_metrics(ppl.fake_results)
        assert ppl_res['ppl_score'] >= 0


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
