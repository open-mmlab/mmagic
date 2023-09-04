# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mmagic.evaluation import (Evaluator, FrechetInceptionDistance,
                               InceptionScore)
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()

fid_loading_str = 'mmagic.evaluation.metrics.fid.FrechetInceptionDistance._load_inception'  # noqa
is_loading_str = 'mmagic.evaluation.metrics.inception_score.InceptionScore._load_inception'  # noqa

loading_mock = MagicMock(return_value=(MagicMock(), 'StyleGAN'))


class TestEvaluator(TestCase):

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
        evaluator = Evaluator(self.metrics)
        self.assertFalse(evaluator.is_ready)

    @patch(is_loading_str, loading_mock)
    @patch(fid_loading_str, loading_mock)
    def test_prepare_metric(self):
        evaluator = Evaluator(self.metrics)
        model = MagicMock()
        model.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()
        with patch('mmagic.evaluation.metrics.fid.prepare_inception_feat'):
            evaluator.prepare_metrics(model, dataloader)
            self.assertTrue(evaluator.is_ready)

        evaluator = Evaluator(self.metrics)
        evaluator.metrics = [MagicMock()]
        evaluator.is_ready = True
        evaluator.prepare_metrics(model, dataloader)
        evaluator.metrics[0].assert_not_called()

    @patch(is_loading_str, loading_mock)
    @patch(fid_loading_str, loading_mock)
    def test_prepare_samplers(self):
        evaluator = Evaluator(self.metrics)

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
        evaluator = Evaluator(cfg)

        # mock metrics
        model = MagicMock()
        model.data_preprocessor.device = 'cpu'

        dataloader = MagicMock()
        dataloader.batch_size = 2

        metric_sampler_list = evaluator.prepare_samplers(model, dataloader)
        self.assertEqual(len(metric_sampler_list), 3)

        # test prepare sampler with metric.need_cond_input = True
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
                need_cond_input=True),
            dict(
                type='InceptionScore',
                fake_nums=10,
                inception_style='pytorch',
                sample_model='ema',
                need_cond_input=True),
            dict(
                type='InceptionScore',
                fake_nums=10,
                inception_style='pytorch',
                sample_model='orig',
                need_cond_input=True),
        ]
        # all metrics (5 groups): [[IS-orig, FID-orig], [TransFID-orig],
        #                          [FID-ema], [FID-cond-ema, IS-cond-ema],
        #                          [IS-cond-orig]]
        evaluator = Evaluator(cfg)

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
        evaluator = Evaluator(self.metrics)
        metrics_mock = [MagicMock(), MagicMock()]

        data_samples = [DataSample(a=1, b=2), dict(c=3, d=4)]

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
        evaluator = Evaluator(self.metrics)

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


class TestNonMetricEvaluator(TestCase):

    def test_init(self):
        evaluator = Evaluator(None)
        self.assertIsNone(evaluator.metrics)

    def test_prepare_metrics(self):
        evaluator = Evaluator(None)
        evaluator.prepare_metrics(None, None)
        self.assertTrue(evaluator.is_ready)

    def test_prepare_samplers(self):
        evaluator = Evaluator(None)
        metric_sampler_list = evaluator.prepare_samplers(None, None)
        self.assertEqual(metric_sampler_list, [[[None], []]])

    def test_process(self):
        evaluator = Evaluator(None)
        evaluator.process(None, None, None)

    def test_evalute(self):
        evaluator = Evaluator(None)
        output = evaluator.evaluate()
        self.assertEqual(output, {'No Metric': 'Nan'})


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
