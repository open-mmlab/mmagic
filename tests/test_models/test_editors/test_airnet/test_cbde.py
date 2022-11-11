# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmedit.models.editors.airnet.cbde import CBDE, ResBlock, ResEncoder


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_cdbe():
    model = CBDE(
        batch_size=1,
        dim=256,
    )

    assert model.__class__.__name__ == 'CBDE'

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


def test_resblock():
    model = ResBlock(in_feat=64, out_feat=128, stride=2)

    assert model.__class__.__name__ == 'ResBlock'

    # prepare data
    inputs = torch.rand(1, 64, 64, 64)
    targets = torch.rand(1, 128, 32, 32)

    # test on cpu
    output = model(inputs)
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
        output = model(inputs)
        assert torch.is_tensor(output)
        assert output.shape == targets.shape


def test_resencoder():
    model = ResEncoder()

    assert model.__class__.__name__ == 'ResEncoder'

    # prepare data
    inputs = torch.rand(1, 3, 64, 64)

    # test on cpu
    feat, out, inter = model(inputs)
    assert torch.is_tensor(feat)
    assert torch.is_tensor(out)
    assert torch.is_tensor(inter)
    assert inter.shape == torch.Size([1, 64, 64, 64])
    assert out.shape == torch.Size([1, 256])
    assert feat.shape == torch.Size([1, 256])

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()

        feat, out, inter = model(inputs)
        assert torch.is_tensor(feat)
        assert torch.is_tensor(out)
        assert torch.is_tensor(inter)
        assert inter.shape == torch.Size([1, 64, 64, 64])
        assert out.shape == torch.Size([1, 256])
        assert feat.shape == torch.Size([1, 256])
