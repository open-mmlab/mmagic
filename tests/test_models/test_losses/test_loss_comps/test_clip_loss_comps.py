# Copyright (c) OpenMMLab. All rights reserved.
import clip
import torch

from mmedit.models.losses import CLIPLossComps


def test_clip_loss():
    clip_loss_comps = CLIPLossComps(
        clip_model=dict(in_size=32),
        data_info=dict(image='fake_imgs', text='descriptions'))

    image = torch.randn(1, 3, 32, 32)
    text = 'Image for test'
    text_inputs = torch.cat([clip.tokenize(text)])
    data_dict = dict(fake_imgs=image, descriptions=text_inputs)
    loss = clip_loss_comps(outputs_dict=data_dict)
    print(loss)
