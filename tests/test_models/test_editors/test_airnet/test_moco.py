# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.airnet.cbde import ResEncoder
from mmedit.models.editors.airnet.moco import MoCo


def test_moco():
    model = MoCo(base_encoder=ResEncoder, dim=256, K=256)

    assert model.__class__.__name__ == 'MoCo'

    # prepare data
    inputs = torch.rand(1, 3, 64, 64)
    targets = torch.rand(1, 64, 64, 64)

    # test on cpu
    output = model(inputs, inputs)
    assert isinstance(output, dict)
    feat = output['feat']
    assert torch.is_tensor(feat)
    inter = output['inter']
    assert torch.is_tensor(inter)
    assert feat.shape == torch.Size([1, 256])
    assert inter.shape == targets.shape
    logits = output['logits']
    labels = output['labels']
    assert torch.is_tensor(logits)
    assert torch.is_tensor(labels)
    assert logits.shape == torch.Size([1, 257])
    assert labels.shape == torch.Size([1])

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
        output = model(inputs, inputs)
        assert isinstance(output, dict)
        feat = output['feat']
        assert torch.is_tensor(feat)
        inter = output['inter']
        assert torch.is_tensor(inter)
        assert feat.shape == torch.Size([1, 256])
        assert inter.shape == targets.shape
        logits = output['logits']
        labels = output['labels']
        assert torch.is_tensor(logits)
        assert torch.is_tensor(labels)
        assert logits.shape == torch.Size([1, 257])
        assert labels.shape == torch.Size([1])
