# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.models.archs import (LoRAWrapper, set_lora, set_lora_disable,
                                 set_lora_enable, set_only_lora_trainable)


class ToyAttn(nn.Module):

    def __init__(self, in_dim, context_dim):
        super().__init__()
        self.to_q = nn.Linear(in_dim, in_dim)
        self.to_k = nn.Linear(context_dim, in_dim)
        self.to_v = nn.Linear(context_dim, in_dim)

    def forward(self, x, context=None):

        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        if context is None:
            context = x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        attn_mask = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
        out = attn_mask @ v
        return out.view(b, h, w, c).permute(0, 3, 1, 2)


class ToySubModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Identity()
        self.attn1 = ToyAttn(4, 3)

    def forward(self, x, context=None):
        x = self.net(x)
        x = self.attn1(x, context)
        return x


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.n1 = ToySubModule()
        self.attn2 = ToyAttn(4, 4)

    def forward(self, x, context=None):
        out = self.n1(x, context)
        out = self.attn2(out)
        return out


@pytest.mark.skipif(
    digit_version(TORCH_VERSION) <= digit_version('1.8.1'),
    reason='get_submodule requires torch >= 1.9.0')
def test_set_lora():
    model = ToyModel()

    img = torch.randn(2, 4, 3, 3)
    context = torch.randn(2, 11, 3)

    config = dict(rank=2, scale=1, target_modules='to_q')
    model: ToyModel = set_lora(model, config, True)
    isinstance(model.attn2.to_q, LoRAWrapper)
    isinstance(model.n1.attn1.to_q, LoRAWrapper)

    out = model(img, context)
    assert out.shape == (2, 4, 3, 3)

    model = ToyModel()
    config = dict(
        rank=2,
        scale=1,
        target_modules=[
            'to_q',
            dict(target_module='.*attn1.to_v', rank=1),
            dict(target_module='to_k', scale=2.5)
        ])
    out_wo_lora = model(img, context)
    set_lora(model, config)

    assert isinstance(model.attn2.to_q, LoRAWrapper)
    assert model.attn2.to_q.scale == 1
    assert model.attn2.to_q.rank == 2
    assert isinstance(model.attn2.to_k, LoRAWrapper)
    assert model.attn2.to_k.scale == 2.5
    assert model.attn2.to_k.rank == 2
    assert isinstance(model.attn2.to_v, nn.Linear)

    assert isinstance(model.n1.attn1.to_q, LoRAWrapper)
    assert model.n1.attn1.to_q.scale == 1
    assert model.n1.attn1.to_q.rank == 2
    assert isinstance(model.n1.attn1.to_k, LoRAWrapper)
    assert model.n1.attn1.to_k.scale == 2.5
    assert model.n1.attn1.to_k.rank == 2
    assert isinstance(model.n1.attn1.to_v, LoRAWrapper)
    assert model.n1.attn1.to_v.scale == 1
    assert model.n1.attn1.to_v.rank == 1

    out_w_lora = model(img, context)
    assert out_w_lora.shape == (2, 4, 3, 3)

    model.n1.attn1.to_v.set_scale(10)
    assert model.n1.attn1.to_v.scale == 10

    # test set onlyu lora trainable
    set_only_lora_trainable(model)
    for n, m in model.named_parameters():
        if 'lora_' in n:
            assert m.requires_grad
        else:
            assert not m.requires_grad

    # test enable and disable
    set_lora_disable(model)
    out_lora_disable = model(img, context)
    assert (out_lora_disable == out_wo_lora).all()
    set_lora_enable(model)
    out_lora_enable = model(img, context)
    assert (out_lora_enable == out_w_lora).all()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
