# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mmedit.evaluation import (FrechetInceptionDistance, GenEvaluator,
                               InceptionScore)
from mmedit.structures import EditDataSample
from mmedit.utils import register_all_modules

register_all_modules()

fid_loading_str = 'mmedit.evaluation.metrics.fid.FrechetInceptionDistance._load_inception'  # noqa
is_loading_str = 'mmedit.evaluation.metrics.inception_score.InceptionScore._load_inception'  # noqa

loading_mock = MagicMock(return_value=(MagicMock(), 'StyleGAN'))


class TestGenEvaluator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.metrics = [
            dict(
                type='InceptionScore',
                fake_nums=10,
                inception_style='pytorch',
                sample_model='orig'),
            dict(
                type='FrechetInceptionDistance',
                fake_nums=11,
                inception_style='pytorch',
                sample_model='orig'),
            dict(type='TransFID', fake_nums=10),
        ]

    @patch(is_loading_str, loading_mock)
    @patch(fid_loading_str, loading_mock)
    def test_init(self):
        evaluator = GenEvaluator(self.metrics)
        self.assertFalse(evaluator.is_ready)

    @patch(is_loading_str, loading_mock)
    @patch(fid_loading_str, loading_mock)
    def test_prepare_metric(self):
        evaluator = GenEvaluator(self.metrics)
        model = MagicMock()
        model.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()
        with patch('mmedit.evaluation.metrics.fid.prepare_inception_feat'):
            evaluator.prepare_metrics(model, dataloader)
            self.assertTrue(evaluator.is_ready)

        evaluator = GenEvaluator(self.metrics)
        evaluator.metrics = [MagicMock()]
        evaluator.is_ready = True
        evaluator.prepare_metrics(model, dataloader)
        evaluator.metrics[0].assert_not_called()

    @patch(is_loading_str, loading_mock)
    @patch(fid_loading_str, loading_mock)
    def test_prepare_samplers(self):
        evaluator = GenEvaluator(self.metrics)

        model = MagicMock()
        model.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()
        dataloader.batch_size = 2

        metric_sampler_list = evaluator.prepare_samplers(model, dataloader)
        self.assertEqual(len(metric_sampler_list), 2)
        for metric_cls in [FrechetInceptionDistance, InceptionScore]:
            self.assertTrue(
                any([
                    isinstance(m, metric_cls)
                    for m in metric_sampler_list[0][0]
                ]))
        self.assertEqual(metric_sampler_list[0][1].max_length, 11)
        self.assertEqual(len(metric_sampler_list[0][1]), 6)

        # test prepare metrics with different `sample_model`
        cfg = deepcopy(self.metrics)
        cfg.append(
            dict(
                type='FrechetInceptionDistance',
                fake_nums=12,
                inception_style='pytorch',
                sample_model='ema'))
        evaluator = GenEvaluator(cfg)

        # mock metrics
        model = MagicMock()
        model.data_preprocessor.device = 'cpu'

        dataloader = MagicMock()
        dataloader.batch_size = 2

        metric_sampler_list = evaluator.prepare_samplers(model, dataloader)
        self.assertEqual(len(metric_sampler_list), 3)

        # test prepare sampler with metric.need_cond = True
        cfg = deepcopy(self.metrics)
        cfg += [
            dict(
                type='FrechetInceptionDistance',
                fake_nums=12,
                inception_style='pytorch',
                sample_model='ema'),
            dict(
                type='FrechetInceptionDistance',
                fake_nums=12,
                inception_style='pytorch',
                sample_model='ema',
                need_cond=True),
            dict(
                type='InceptionScore',
                fake_nums=10,
                inception_style='pytorch',
                sample_model='ema',
                need_cond=True),
            dict(
                type='InceptionScore',
                fake_nums=10,
                inception_style='pytorch',
                sample_model='orig',
                need_cond=True),
        ]
        # all metrics (5 groups): [[IS-orig, FID-orig], [TransFID-orig],
        #                          [FID-ema], [FID-cond-ema, IS-cond-ema],
        #                          [IS-cond-orig]]
        evaluator = GenEvaluator(cfg)

        # mock metrics
        model = MagicMock()
        model.data_preprocessor.device = 'cpu'

        dataloader = MagicMock()
        dataloader.batch_size = 2

        metric_sampler_list = evaluator.prepare_samplers(model, dataloader)
        self.assertEqual(len(metric_sampler_list), 5)

    @patch(is_loading_str, loading_mock)
    @patch(fid_loading_str, loading_mock)
    def test_process(self):
        evaluator = GenEvaluator(self.metrics)
        metrics_mock = [MagicMock(), MagicMock()]

        data_samples = [EditDataSample(a=1, b=2), dict(c=3, d=4)]

        # NOTE: data_batch is not used in evaluation
        evaluator.process(
            data_batch=None, data_samples=data_samples, metrics=metrics_mock)

        for metric in metrics_mock:
            metric.process.assert_called_with(None, [{
                'a': 1,
                'b': 2
            }, {
                'c': 3,
                'd': 4
            }])

    @patch(is_loading_str, loading_mock)
    @patch(fid_loading_str, loading_mock)
    def test_evaluate(self):
        evaluator = GenEvaluator(self.metrics)

        # mock metrics
        metric_mock1, metric_mock2 = MagicMock(), MagicMock()
        metric_mock1.evaluate = MagicMock(return_value=dict(m1=233))
        metric_mock2.evaluate = MagicMock(return_value=dict(m2=42))
        evaluator.metrics = [metric_mock1, metric_mock2]

        res = evaluator.evaluate()
        self.assertEqual(res, dict(m1=233, m2=42))

        # test raise value error with duplicate keys
        metric_mock3, metric_mock4 = MagicMock(), MagicMock()
        metric_mock3.evaluate = MagicMock(return_value=dict(m=3))
        metric_mock4.evaluate = MagicMock(return_value=dict(m=4))
        evaluator.metrics = [metric_mock3, metric_mock4]
        with self.assertRaises(ValueError):
            evaluator.evaluate()
