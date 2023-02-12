# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import pytest
from mmengine.evaluator import Evaluator

from mmedit.engine.runner.loop_utils import (is_evaluator,
                                             update_and_check_evaluator)
from mmedit.evaluation import GenEvaluator


def test_is_evaluator():
    evaluator = dict(type='GenEvaluator', metrics=[dict(type='PSNR')])
    assert is_evaluator(evaluator)

    evaluator = [dict(type='PSNR'), dict(type='SSIM')]
    assert is_evaluator(evaluator)

    evaluator = MagicMock(spec=Evaluator)
    assert is_evaluator(evaluator)

    evaluator = 'SSIM'
    assert not is_evaluator(evaluator)

    evaluator = [dict(metrics='PSNR'), dict(metrics='SSIM')]
    assert not is_evaluator(evaluator)

    evaluator = dict(type='PSNR')
    assert not is_evaluator(evaluator)


def test_update_and_check_evaluator():

    evaluator = MagicMock(spec=Evaluator)
    assert evaluator == update_and_check_evaluator(evaluator)

    evaluator = MagicMock(spec=GenEvaluator)
    assert evaluator == update_and_check_evaluator(evaluator)

    evaluator = [dict(type='PSNR'), dict(type='SSIM')]
    evaluator = update_and_check_evaluator(evaluator)
    assert isinstance(evaluator, dict)
    assert evaluator['type'] == 'GenEvaluator'

    evaluator = 'this is wrong'
    with pytest.raises(AssertionError):
        update_and_check_evaluator(evaluator)

    evaluator = dict(metrics=[dict(type='PSNR')])
    evaluator = update_and_check_evaluator(evaluator)
    assert 'type' in evaluator
    assert evaluator['type'] == 'GenEvaluator'

    evaluator = dict(type='Evaluator', metrics=[dict(type='PSNR')])
    evaluator = update_and_check_evaluator(evaluator)
    assert evaluator['type'] == 'Evaluator'

    evaluator = dict(type='GenEvaluator', metrics=[dict(type='PSNR')])
    evaluator = update_and_check_evaluator(evaluator)
    assert evaluator['type'] == 'GenEvaluator'
