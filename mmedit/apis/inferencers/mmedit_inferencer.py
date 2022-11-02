# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np
from typing import Dict, List, Optional, Union


from mmedit.utils import ConfigType
from .base_mmedit_inferencer import (BaseMMEditInferencer, InputsType, PredType,
                                    ResType)
from .conditional_inferencer import ConditionalInferencer
from .unconditional_inferencer import UnconditionalInferencer

class MMEditInferencer(BaseMMEditInferencer):

    def __init__(self,
                 type: Optional[str] = None,
                 config: Optional[Union[ConfigType, str]] = None,
                 ckpt: Optional[str] = None,
                 device: Optional[str] = None,
                 **kwargs) -> None:

        self.type = type
        self.visualizer = None
        self.base_params = self._dispatch_kwargs(*kwargs)
        self.num_visualized_imgs = 0
        if self.type == 'conditional':
            self.inferencer = ConditionalInferencer(config, ckpt, device)
        elif self.type == 'unconditional':
            self.inferencer = UnconditionalInferencer(config, ckpt, device)
        else:
            raise ValueError(f'Unknown inferencer type: {self.type}')
    
    def __call__(self, img: InputsType, label: InputsType, **kwargs) -> Union[Dict, List[Dict]]:
        return self.inferencer(img, label, **kwargs)

