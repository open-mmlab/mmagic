# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from mmengine.runner import Runner

from mmagic.datasets import BasicImageDataset
from mmagic.datasets.transforms import PackInputs
from mmagic.evaluation import PrecisionAndRecall
from mmagic.models import LSGAN, DataPreprocessor
from mmagic.models.editors.dcgan import DCGANGenerator
from mmagic.utils import register_all_modules

register_all_modules()


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


class TestPR:

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(type='LoadImageFromFile', key='gt'),
            dict(type='Resize', keys='gt', scale=(128, 128)),
            PackInputs()
        ]
        dataset = BasicImageDataset(
            data_root='tests/data/image/img_root',
            pipeline=pipeline,
            data_prefix=dict(gt=''),
            test_mode=True,
            recursive=True)
        cls.dataloader = Runner.build_dataloader(
            dict(
                batch_size=2,
                dataset=dataset,
                sampler=dict(type='DefaultSampler')))
        gan_data_preprocessor = DataPreprocessor()
        generator = DCGANGenerator(128, noise_size=10, base_channels=20)
        cls.module = LSGAN(
            generator, data_preprocessor=gan_data_preprocessor)  # noqa

        cls.mock_vgg_pytorch = MagicMock(
            return_value=(vgg_mock('PyTorch'), 'False'))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires cuda')  # noqa
    def test_pr_cuda(self):
        pr = PrecisionAndRecall(10, sample_model='orig', auto_save=False)
        self.module.cuda()
        sampler = pr.get_metric_sampler(self.module, self.dataloader, [pr])
        pr.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            predictions = [pred.to_dict() for pred in predictions]
            pr.process(data_batch, predictions)
        pr_score = pr.compute_metrics(pr.fake_results)
        print(pr_score)
        assert pr_score['precision'] >= 0 and pr_score['recall'] >= 0

    def test_pr_cpu(self):
        with patch.object(PrecisionAndRecall, '_load_vgg',
                          self.mock_vgg_pytorch):
            pr = PrecisionAndRecall(10, sample_model='orig', auto_save=False)
        sampler = pr.get_metric_sampler(self.module, self.dataloader, [pr])
        pr.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            data_samples = self.module.test_step(data_batch)
            data_samples = [pred.to_dict() for pred in data_samples]
            pr.process(data_batch, data_samples)
        pr_score = pr.evaluate()
        print(pr_score)
        assert pr_score['precision'] >= 0 and pr_score['recall'] >= 0


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
