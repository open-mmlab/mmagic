# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.runner import Runner
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version

from mmedit.datasets import BasicImageDataset
from mmedit.datasets.transforms import PackEditInputs
from mmedit.evaluation import Equivariance
from mmedit.models import EditDataPreprocessor, StyleGAN3
from mmedit.models.editors.stylegan3 import StyleGAN3Generator
from mmedit.utils import register_all_modules

register_all_modules()


def process_fn(data_batch, predictions):

    _predictions = []
    for pred in predictions:
        _predictions.append(pred.to_dict())
    return data_batch, _predictions


@pytest.mark.skipif(
    digit_version(TORCH_VERSION) < digit_version('1.8.0'),
    reason='version limitation')
class TestEquivariance:

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(type='LoadImageFromFile', key='img'),
            dict(type='Resize', scale=(64, 64)),
            PackEditInputs()
        ]
        dataset = BasicImageDataset(
            data_root='./tests/data/image/img_root',
            pipeline=pipeline,
            test_mode=True)
        cls.dataloader = Runner.build_dataloader(
            dict(
                batch_size=2,
                dataset=dataset,
                sampler=dict(type='DefaultSampler')))
        gan_data_preprocessor = EditDataPreprocessor()
        generator = StyleGAN3Generator(64, 8, 3, noise_size=8)
        cls.module = StyleGAN3(
            generator, data_preprocessor=gan_data_preprocessor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    @torch.no_grad()
    def test_eq_cuda(self):
        eq = Equivariance(
            2,
            eq_cfg=dict(
                compute_eqt_int=True, compute_eqt_frac=True, compute_eqr=True),
            sample_mode='orig')
        self.module.cuda()
        sampler = eq.get_metric_sampler(self.module, self.dataloader, [eq])
        eq.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            eq.process(_data_batch, _predictions)
        eq_res = eq.compute_metrics(eq.fake_results)
        isinstance(eq_res['eqt_int'], float) and isinstance(
            eq_res['eqt_frac'], float) and isinstance(eq_res['eqr'], float)

    @torch.no_grad()
    def test_eq_cpu(self):
        eq = Equivariance(
            2,
            eq_cfg=dict(
                compute_eqt_int=True, compute_eqt_frac=True, compute_eqr=True),
            sample_mode='orig')
        sampler = eq.get_metric_sampler(self.module, self.dataloader, [eq])
        eq.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            eq.process(_data_batch, _predictions)
        eq_res = eq.compute_metrics(eq.fake_results)
        isinstance(eq_res['eqt_int'], float) and isinstance(
            eq_res['eqt_frac'], float) and isinstance(eq_res['eqr'], float)
