# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from mmengine.runner import Runner

from mmedit.datasets import BasicImageDataset
from mmedit.datasets.transforms import PackEditInputs
from mmedit.evaluation import PerceptualPathLength
from mmedit.models import LSGAN, GenDataPreprocessor
from mmedit.models.editors.stylegan2 import StyleGAN2Generator


def process_fn(data_batch, predictions):

    _predictions = []
    for pred in predictions:
        _predictions.append(pred.to_dict())
    return data_batch, _predictions


class vgg_pytorch_classifier(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.randn(x.shape[0], 4096)


class vgg_mock(nn.Module):

    def __init__(self, style):
        super().__init__()
        self.classifier = nn.Sequential(nn.Identity(), nn.Identity(),
                                        nn.Identity(),
                                        vgg_pytorch_classifier())
        self.style = style

    def forward(self, x, *args, **kwargs):
        if self.style.upper() == 'STYLEGAN':
            return torch.randn(x.shape[0], 4096)
        else:  # torch
            return torch.randn(x.shape[0], 7 * 7 * 512)


class TestPPL:

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(type='LoadImageFromFile', key='img'),
            dict(type='Resize', scale=(64, 64)),
            PackEditInputs()
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
        gan_data_preprocessor = GenDataPreprocessor()
        generator = StyleGAN2Generator(64, 8)
        cls.module = LSGAN(generator, data_preprocessor=gan_data_preprocessor)

        cls.mock_vgg_pytorch = MagicMock(
            return_value=(vgg_mock('PyTorch'), 'False'))

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
