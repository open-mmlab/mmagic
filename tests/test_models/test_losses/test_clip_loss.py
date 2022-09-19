# Copyright (c) OpenMMLab. All rights reserved.
import clip
import torch

from mmedit.models.losses import CLIPLoss


def test_clip_loss():
    clip_loss = CLIPLoss(clip_model=dict(in_size=32))

    image = torch.randn(1, 3, 32, 32)
    text = 'Image for test'
    text_inputs = torch.cat([clip.tokenize(text)])
    loss = clip_loss(image, text_inputs)
    print(loss)
