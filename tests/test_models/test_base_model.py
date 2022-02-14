# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest.mock import patch

import pytest
import torch

from mmedit.models import BaseModel


class TestBaseModel(unittest.TestCase):

    @patch.multiple(BaseModel, __abstractmethods__=set())
    def test_parse_losses(self):
        self.base_model = BaseModel()
        with pytest.raises(TypeError):
            losses = dict(loss=0.5)
            self.base_model.parse_losses(losses)

        a_loss = [torch.randn(5, 5), torch.randn(5, 5)]
        b_loss = torch.randn(5, 5)
        losses = dict(a_loss=a_loss, b_loss=b_loss)
        r_a_loss = sum(_loss.mean() for _loss in a_loss)
        r_b_loss = b_loss.mean()
        r_loss = [r_a_loss, r_b_loss]
        r_loss = sum(r_loss)

        loss, log_vars = self.base_model.parse_losses(losses)

        assert r_loss == loss
        assert set(log_vars.keys()) == set(['a_loss', 'b_loss', 'loss'])
        assert log_vars['a_loss'] == r_a_loss
        assert log_vars['b_loss'] == r_b_loss
        assert log_vars['loss'] == r_loss
