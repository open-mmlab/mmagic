# Copyright (c) OpenMMLab. All rights reserved.
import clip
import open_clip
import torch.nn as nn

from mmedit.registry import MODELS


@MODELS.register_module()
class ClipWrapper(nn.Module):

    def __init__(self, clip_type, *args, **kwargs):
        super().__init__()
        self.clip_type = clip_type
        assert clip_type in ['clip', 'open_clip']
        if clip_type == 'clip':
            self.model, _ = clip.load(*args, **kwargs)
        elif clip_type == 'open_clip':
            self.model = open_clip.create_model(*args, **kwargs)
        self.model.eval().requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
