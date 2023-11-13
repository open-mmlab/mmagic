# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.optim import OptimWrapper
from torch import nn
from torch.optim import Adam

from mmagic.models import BaseEditModel, DataPreprocessor
from mmagic.models.losses import L1Loss
from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


@MODELS.register_module()
class ToyBaseModel(nn.Module):
    """An example of interpolate network for testing BasicInterpolator."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        return self.layer(x)

    def init_weights(self, pretrained=None):
        pass


def test_base_edit_model():

    model = BaseEditModel(
        generator=dict(type='ToyBaseModel'),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        data_preprocessor=DataPreprocessor())

    assert model.__class__.__name__ == 'BaseEditModel'
    assert isinstance(model.generator, ToyBaseModel)
    assert isinstance(model.pixel_loss, L1Loss)

    optimizer = Adam(model.generator.parameters(), lr=0.001)
    optim_wrapper = OptimWrapper(optimizer)

    # prepare data
    inputs = torch.rand(1, 3, 20, 20)
    target = torch.rand(3, 20, 20)
    data_sample = DataSample(gt_img=target)
    data = dict(inputs=inputs, data_samples=[data_sample])

    # train
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars['loss'], torch.Tensor)
    save_loss = log_vars['loss']
    log_vars = model.train_step(data, optim_wrapper)
    log_vars = model.train_step(data, optim_wrapper)
    assert save_loss > log_vars['loss']

    # val
    output = model.val_step(data)
    assert output[0].output.pred_img.shape == (3, 20, 20)

    # feat
    output = model(torch.rand(1, 3, 20, 20), mode='tensor')
    assert output.shape == (1, 3, 20, 20)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
