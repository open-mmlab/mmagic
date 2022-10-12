# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Dict, List, Optional, Tuple, Union

import torch
from torchvision.utils import save_image
from mmengine.model import BaseModel
from mmengine.config import Config, ConfigDict

from mmedit.structures import EditDataSample, PixelData
from mmedit.registry import MODELS


class BaseColorization(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 data_preprocessor: Union[dict, Config],
                 loss,
                 init_cfg: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None):

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.loss = MODELS.build(loss)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[Union[list, torch.Tensor]] = None,
                mode: str = 'tensor',
                **kwargs):

        if mode == 'tensor':
            return self.forward_tensor(inputs, data_samples, **kwargs)

        elif mode == 'predict':
            predictions = self.forward_test(inputs, data_samples, **kwargs)
            predictions = self.convert_to_datasample(data_samples, predictions)
            return predictions

        elif mode == 'loss':
            return self.forward_train(inputs, data_samples, **kwargs)

    def forward_train(self, *args, **kwargs):
        pass

    def forward_test(self, input, data_samples, **kwargs):
        pass

    def train_step(self, data_batch, optimizer):
        pass

    def init_weights(self):
        pass

    def save_visualization(self, img, filename):
        save_image(img, filename)
