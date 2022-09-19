# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmedit.models.utils import gather_log_vars


class TestGatherLogVars(TestCase):

    def test(self):
        log_dict_list = [
            dict(loss=torch.Tensor([2.33]), loss_disc=torch.Tensor([1.14514]))
        ]
        self.assertDictEqual(log_dict_list[0], gather_log_vars(log_dict_list))

        log_dict_list = [
            dict(loss=torch.Tensor([2]), loss_disc=torch.Tensor([2])),
            dict(loss=torch.Tensor([3]), loss_disc=torch.Tensor([5]))
        ]
        self.assertDictEqual(
            dict(loss=torch.Tensor([2.5]), loss_disc=torch.Tensor([3.5])),
            gather_log_vars(log_dict_list))

        # test raise error
        with self.assertRaises(AssertionError):
            gather_log_vars([dict(a=1), dict(b=2)])
