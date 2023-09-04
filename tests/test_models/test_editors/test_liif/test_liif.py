# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmagic.models import LIIF, DataPreprocessor
from mmagic.structures import DataSample


def test_liif():

    model = LIIF(
        generator=dict(
            type='LIIFEDSRNet',
            encoder=dict(
                type='EDSRNet',
                in_channels=3,
                out_channels=3,
                mid_channels=4,
                num_blocks=2),
            imnet=dict(
                type='MLPRefiner',
                in_dim=64,
                out_dim=3,
                hidden_list=[4, 4, 4, 4]),
            local_ensemble=True,
            feat_unfold=True,
            cell_decode=True,
            eval_bsize=64),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        data_preprocessor=DataPreprocessor(
            mean=[0.4488 * 255, 0.4371 * 255, 0.4040 * 255],
            std=[255., 255., 255.],
            # input_view=(-1, 1, 1),
            # output_view=(1, -1)
        ))

    # test attributes
    assert model.__class__.__name__ == 'LIIF'

    # prepare data
    inputs = torch.rand(1, 3, 8, 8)
    data_sample = DataSample(
        metainfo=dict(coord=torch.rand(256, 2), cell=torch.rand(256, 2)))
    data_sample.gt_img = torch.rand(256, 3)
    data = dict(inputs=inputs, data_samples=[data_sample])

    optimizer = Adam(model.generator.parameters(), lr=0.001)
    optim_wrapper = OptimWrapper(optimizer)

    # train
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars['loss'], torch.Tensor)
    save_loss = log_vars['loss']
    log_vars = model.train_step(data, optim_wrapper)
    log_vars = model.train_step(data, optim_wrapper)
    assert save_loss > log_vars['loss']

    # val
    predictions = model.val_step(data)
    # predictions = predictions.split()
    assert isinstance(predictions, List)
    assert len(predictions) == 1
    assert isinstance(predictions[0], DataSample)
    assert isinstance(predictions[0].output.pred_img.data, torch.Tensor)
    assert predictions[0].output.pred_img.shape == (3, 16, 16)

    # feat
    output = model(
        torch.rand(1, 3, 8, 8), DataSample.stack([data_sample]), mode='tensor')
    assert output.shape == (1, 256, 3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
