# Copyright (c) OpenMMLab. All rights reserved.
from ..base import BaseModel
from ..registry import MODELS


@MODELS.register_module()
class BaseColorization(BaseModel):

    def initialize(self):
        pass

    def forward(self, img, test_mode=True, **kwargs):
        if test_mode:
            return self.forward_test(img, **kwargs)
        return self.forward_train(img, **kwargs)

    def forward_train(self, *args, **kwargs):
        pass

    def forward_test(self, imgs, **kwargs):
        pass

    def train_step(self, data_batch, optimizer):
        pass

    def init_weights(self):
        pass
