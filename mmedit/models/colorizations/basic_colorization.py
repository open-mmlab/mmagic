# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torchvision.utils import save_image

from ..base import BaseModel
from ..builder import build_loss
from ..registry import MODELS


@MODELS.register_module()
class BaseColorization(BaseModel):

    def __init__(
        self,
        ngf,
        output_nc,
        # avg_loss_alpha,
        ab_norm,
        l_norm,
        l_cent,
        sample_Ps,
        mask_cent,
        init_type,
        fusion_weight_path,
        which_direction,
        loss,
        ab_max=110.,
        ab_quant=10.,
        gpu_ids='0',
        train_cfg=None,
        test_cfg=None,
        pretrained=None):
        super().__init__()
        # BaseColorization.initialize(self)
        self.ngf = ngf
        self.output_nc = output_nc
        # self.avg_loss_alpha = avg_loss_alpha
        self.ab_norm = ab_norm
        self.l_norm = l_norm
        self.l_cent = l_cent
        self.sample_Ps = sample_Ps
        self.mask_cent = mask_cent
        self.init_type = init_type
        if loss:
            self.loss = build_loss(loss)
        self.fusion_weight_path = fusion_weight_path
        self.which_direction = which_direction
        self.ab_max = ab_max
        self.ab_quant = ab_quant
        self.model_names = ['G', 'GF']
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # load/define networks

    def forward(self, test_mode=True, **kwargs):
        if test_mode:
            return self.forward_test(**kwargs)
        return self.forward_train(**kwargs)

    def forward_train(self, *args, **kwargs):
        pass

    def forward_test(self, imgs, **kwargs):
        pass

    def train_step(self, data_batch, optimizer):
        pass

    def init_weights(self):
        pass

    def save_visualization(self, img, filename):
        save_image(img, filename)
