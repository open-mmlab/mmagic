# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from unittest.mock import MagicMock

import pytest
import torch

from mmagic.models.editors.controlnet.controlnet_utils import change_base_model


def make_state_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = torch.FloatTensor([v])
    return new_d


def check_state_dict(s1, s2):
    # s1, s2 = m1.state_dict(), m2.state_dict()
    assert s1.keys() == s2.keys()
    for k in s1.keys():
        assert (s1[k] == s2[k]).all()


def parameters(model):
    state_dict = model.state_dict()
    for v in state_dict.values():
        yield v


def test_change_base_model():
    control_state_dict = make_state_dict(dict(k1=1, k2=2, k3=3))
    target_control_state_dict = make_state_dict(dict(k1=1.5, k2=2.5, k3=3))

    base_state_dict = make_state_dict(dict(k1=2, k2=3))
    curr_state_dict = make_state_dict(dict(k1=2.5, k2=3.5))

    controlnet = MagicMock()
    basemodel = MagicMock()
    currmodel = MagicMock()
    controlnet.parameters = MagicMock(return_value=parameters(controlnet))
    basemodel.parameters = MagicMock(return_value=parameters(basemodel))
    currmodel.parameters = MagicMock(return_value=parameters(currmodel))
    controlnet.state_dict = MagicMock(return_value=control_state_dict)
    basemodel.state_dict = MagicMock(return_value=base_state_dict)
    currmodel.state_dict = MagicMock(return_value=curr_state_dict)

    change_base_model(controlnet, currmodel, basemodel)
    check_state_dict(controlnet.state_dict(), target_control_state_dict)

    # test save
    control_state_dict = make_state_dict(dict(k1=1, k2=2, k3=3))
    controlnet.state_dict = MagicMock(return_value=control_state_dict)
    save_path = osp.abspath(
        osp.join(__file__, '../../../../data/out', 'test.pth'))
    change_base_model(controlnet, currmodel, basemodel, save_path=save_path)
    assert os.path.isfile(save_path)
    control_state_dict_loaded = torch.load(save_path)
    check_state_dict(control_state_dict_loaded, target_control_state_dict)
    os.remove(save_path)

    # test error
    wrong_base_state_dict = make_state_dict(dict(k1=2, k2=[3, 5]))
    wrong_model = MagicMock()
    wrong_model.state_dict = MagicMock(return_value=wrong_base_state_dict)
    with pytest.raises(Exception):
        change_base_model(controlnet, currmodel, wrong_model)
    # change_base_model(controlnet, currmodel, wrong_base_state_dict)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
