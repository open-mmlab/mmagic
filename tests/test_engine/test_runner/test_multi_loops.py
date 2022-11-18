# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmedit.engine import MultiTestLoop, MultiValLoop


def test_multi_val_loop():
    runner = Mock()
    dataloader = [dict(), dict()]
    evaluator = [[dict(), dict()], [dict(), dict()]]
    loop = MultiValLoop(runner, dataloader, evaluator)
    evaluator = Mock()
    evaluator.evaluate = Mock(return_value={'metric': 0})
    loop.evaluators = [evaluator]
    dataloader = Mock(dataset=[0])
    dataloader.__iter__ = lambda s: iter([])
    loop.dataloaders = [dataloader, dataloader]
    loop.run()


def test_multi_test_loop():
    runner = Mock()
    dataloader = [dict(), dict()]
    evaluator = [[dict(), dict()], [dict(), dict()]]
    loop = MultiTestLoop(runner, dataloader, evaluator)
    evaluator = Mock()
    evaluator.evaluate = Mock(return_value={'metric': 0})
    loop.evaluators = [evaluator]
    dataloader = Mock(dataset=[0])
    dataloader.__iter__ = lambda s: iter([])
    loop.dataloaders = [dataloader, dataloader]
    loop.run()
