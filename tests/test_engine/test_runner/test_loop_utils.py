# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import pytest
from mmengine.evaluator import Evaluator as BaseEvaluator

from mmagic.engine.runner.loop_utils import (is_evaluator,
                                             update_and_check_evaluator)
from mmagic.evaluation import Evaluator


def test_is_evaluator():
    evaluator = dict(type='Evaluator', metrics=[dict(type='PSNR')])
    assert is_evaluator(evaluator)

    evaluator = [dict(type='PSNR'), dict(type='SSIM')]
    assert is_evaluator(evaluator)

    evaluator = MagicMock(spec=BaseEvaluator)
    assert is_evaluator(evaluator)

    evaluator = 'SSIM'
    assert not is_evaluator(evaluator)

    evaluator = [dict(metrics='PSNR'), dict(metrics='SSIM')]
    assert not is_evaluator(evaluator)

    evaluator = dict(type='PSNR')
    assert not is_evaluator(evaluator)


def test_update_and_check_evaluator():

    evaluator = MagicMock(spec=BaseEvaluator)
    assert evaluator == update_and_check_evaluator(evaluator)

    evaluator = MagicMock(spec=Evaluator)
    assert evaluator == update_and_check_evaluator(evaluator)

    evaluator = [dict(type='PSNR'), dict(type='SSIM')]
    evaluator = update_and_check_evaluator(evaluator)
    assert isinstance(evaluator, dict)
    assert evaluator['type'] == 'Evaluator'

    evaluator = 'this is wrong'
    with pytest.raises(AssertionError):
        update_and_check_evaluator(evaluator)

    evaluator = dict(metrics=[dict(type='PSNR')])
    evaluator = update_and_check_evaluator(evaluator)
    assert 'type' in evaluator
    assert evaluator['type'] == 'Evaluator'

    evaluator = dict(type='Evaluator', metrics=[dict(type='PSNR')])
    evaluator = update_and_check_evaluator(evaluator)
    assert evaluator['type'] == 'Evaluator'

    evaluator = dict(type='Evaluator', metrics=[dict(type='PSNR')])
    evaluator = update_and_check_evaluator(evaluator)
    assert evaluator['type'] == 'Evaluator'

    evaluator = dict(type='Evaluator')
    evaluator = update_and_check_evaluator(evaluator)
    assert evaluator['metrics'] is None


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
