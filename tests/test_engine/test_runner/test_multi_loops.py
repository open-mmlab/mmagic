# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

from mmengine.evaluator import Evaluator as BaseEvaluator

from mmagic.engine import MultiTestLoop, MultiValLoop
from mmagic.evaluation import Evaluator


def build_dataloader(loader, **kwargs):
    if isinstance(loader, dict):
        dataset = MagicMock()
        dataloader = MagicMock()
        dataloader.dataset = dataset
        return dataloader
    else:
        return loader


def build_metrics(metrics):
    if isinstance(metrics, dict):
        return [MagicMock(**metrics)]
    elif isinstance(metrics, list):
        return [MagicMock(**metric) for metric in metrics]
    else:
        raise ValueError('Unsupported metrics type in MockRunner.')


def build_evaluator(evaluator):
    if isinstance(evaluator, BaseEvaluator):
        return evaluator

    if isinstance(evaluator, dict):

        # a dirty way to check Evaluator type
        if 'type' in evaluator and evaluator['type'] == 'Evaluator':
            spec = Evaluator
        else:
            spec = BaseEvaluator

        # if `metrics` in dict keys, it means to build customized evalutor
        if 'metrics' in evaluator:
            evaluator_ = MagicMock(spec=spec)
            evaluator_.metrics = build_metrics(evaluator['metrics'])
            return evaluator_
        # otherwise, default evalutor will be built
        else:
            evaluator_ = MagicMock(spec=spec)
            evaluator_.metrics = build_metrics(evaluator)
            return evaluator_

    elif isinstance(evaluator, list):
        # use the default `Evaluator`
        evaluator_ = MagicMock(spec=Evaluator)
        evaluator_.metrics = build_metrics(evaluator)
        return evaluator_
    else:
        raise TypeError(
            'evaluator should be one of dict, list of dict, and Evaluator'
            f', but got {evaluator}')


def build_mock_runner():
    runner = MagicMock()
    runner.build_evaluator = build_evaluator
    runner.build_dataloader = build_dataloader
    return runner


class TestLoop(TestCase):

    def _test_init(self, is_val):
        LOOP_CLS = MultiValLoop if is_val else MultiTestLoop

        # test init with single evaluator
        runner = build_mock_runner()
        dataloaders = MagicMock()
        evaluators = [dict(prefix='m1'), dict(prefix='m2')]
        loop = LOOP_CLS(runner, dataloaders, evaluators)
        self.assertEqual(len(loop.evaluators), 1)
        self.assertIsInstance(loop.evaluators[0], Evaluator)
        self.assertEqual(loop.evaluators[0].metrics[0].prefix, 'm1')
        self.assertEqual(loop.evaluators[0].metrics[1].prefix, 'm2')

        # test init with single evaluator and dataloader is list
        runner = build_mock_runner()
        dataloaders = [MagicMock()]
        evaluators = dict(
            type='Evaluator', metrics=[dict(prefix='m1'),
                                       dict(prefix='m2')])
        loop = LOOP_CLS(runner, dataloaders, evaluators)
        self.assertEqual(len(loop.evaluators), 1)
        self.assertIsInstance(loop.evaluators[0], BaseEvaluator)
        self.assertEqual(loop.evaluators[0].metrics[0].prefix, 'm1')
        self.assertEqual(loop.evaluators[0].metrics[1].prefix, 'm2')

        # test init with multi evaluators
        runner = build_mock_runner()
        dataloaders = [MagicMock(), MagicMock()]
        evaluators = [
            dict(
                type='Evaluator',
                metrics=[dict(prefix='m1'),
                         dict(prefix='m2')]),
            dict(metrics=dict(prefix='m3'))
        ]
        loop = LOOP_CLS(runner, dataloaders, evaluators)
        self.assertEqual(len(loop.evaluators), 2)
        self.assertIsInstance(loop.evaluators[0], BaseEvaluator)
        self.assertIsInstance(loop.evaluators[1], Evaluator)
        self.assertEqual(loop.evaluators[0].metrics[0].prefix, 'm1')
        self.assertEqual(loop.evaluators[0].metrics[1].prefix, 'm2')
        self.assertEqual(loop.evaluators[1].metrics[0].prefix, 'm3')

        # test call total length before self.run
        self.assertEqual(loop.total_length, 0)

    def test_init(self):
        self._test_init(True)  # val
        self._test_init(False)  # test

    def _test_run(self, is_val):
        # since we have tested init, we direct use predefined mock object to
        # test run function
        LOOP_CLS = MultiValLoop if is_val else MultiTestLoop

        # test single evaluator
        runner = build_mock_runner()

        dataloader = MagicMock()
        dataloader.batch_size = 3

        metric1, metric2, metric3 = MagicMock(), MagicMock(), MagicMock()

        evaluator = MagicMock(spec=Evaluator)
        evaluator.prepare_metrics = MagicMock()
        evaluator.prepare_samplers = MagicMock(
            return_value=[[[metric1, metric2],
                           [dict(inputs=1), dict(
                               inputs=2)]], [[metric3], [dict(inputs=4)]]])

        loop = LOOP_CLS(
            runner=runner, dataloader=dataloader, evaluator=evaluator)
        assert len(loop.evaluators) == 1
        assert loop.evaluators[0] == evaluator

        # test run
        loop.run()

        assert loop.total_length == 3
        call_args_list = evaluator.call_args_list
        for idx, call_args in enumerate(call_args_list):
            if idx == 0:
                inputs = dict(inputs=1)
            elif idx == 1:
                inputs = dict(inputs=2)
            else:
                inputs = dict(inputs=4)
            assert call_args[1] == inputs

        # test multi evaluator
        runner = build_mock_runner()
        dataloader = MagicMock()
        dataloader.batch_size = 3

        metric11, metric12, metric13 = MagicMock(), MagicMock(), MagicMock()
        metric21 = MagicMock()
        evaluator1 = MagicMock(spec=Evaluator)
        evaluator1.prepare_metrics = MagicMock()
        evaluator1.prepare_samplers = MagicMock(
            return_value=[[[metric11, metric12],
                           [dict(inputs=1), dict(
                               inputs=2)]], [[metric13], [dict(inputs=4)]]])
        evaluator2 = MagicMock(spec=Evaluator)
        evaluator2.prepare_metrics = MagicMock()
        evaluator2.prepare_samplers = MagicMock(
            return_value=[[[metric21], [dict(inputs=3)]]])
        loop = LOOP_CLS(
            runner=runner,
            dataloader=[dataloader, dataloader],
            evaluator=[evaluator1, evaluator2])
        assert len(loop.evaluators) == 2
        assert loop.evaluators[0] == evaluator1
        assert loop.evaluators[1] == evaluator2

        loop.run()

        assert loop.total_length == 4
        call_args_list = evaluator1.call_args_list
        for idx, call_args in enumerate(call_args_list):
            if idx == 0:
                inputs = dict(inputs=1)
            elif idx == 1:
                inputs = dict(inputs=2)
            else:
                inputs = dict(inputs=4)
            assert call_args[1] == inputs
        call_args_list = evaluator2.call_args_list
        for idx, call_args in enumerate(call_args_list):
            if idx == 0:
                inputs = dict(inputs=3)
            assert call_args[1] == inputs

    def test_run(self):
        self._test_run(True)  # val
        self._test_run(False)  # test


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
