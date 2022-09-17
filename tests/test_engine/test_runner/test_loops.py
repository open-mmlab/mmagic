# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import pytest
from mmengine.evaluator import Evaluator

from mmedit.engine import GenTestLoop, GenValLoop
from mmedit.utils import register_all_modules

register_all_modules()


@pytest.mark.parametrize('LOOP_CLS', [GenValLoop, GenTestLoop])
def test_loops(LOOP_CLS):
    runner = MagicMock()

    dataloader = MagicMock()
    dataloader.batch_size = 3

    metric1, metric2, metric3 = MagicMock(), MagicMock(), MagicMock()

    evaluator = MagicMock(spec=Evaluator)
    evaluator.prepare_metrics = MagicMock()
    evaluator.prepare_samplers = MagicMock(
        return_value=[[[metric1, metric2], [dict(
            inputs=1), dict(inputs=2)]], [[metric3], [dict(inputs=4)]]])

    # test init
    loop = LOOP_CLS(runner=runner, dataloader=dataloader, evaluator=evaluator)
    assert loop.evaluator == evaluator

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
