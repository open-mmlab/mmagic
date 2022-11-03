# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

from mmedit.utils import ConfigType
from .base_mmedit_inferencer import BaseMMEditInferencer
from .conditional_inferencer import ConditionalInferencer
from .unconditional_inferencer import UnconditionalInferencer
from .matting_inferencer import MattingInferencer
from .inpainting_inferencer import InpaintingInferencer
from .translation_inferencer import TranslationInferencer
from .restoration_inferencer import RestorationInferencer
from .video_restoration_inferencer import VideoRestorationInferencer
from .video_interpolation_inferencer import VideoInterpolationInferencer

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
        elif self.type == 'matting':
            self.inferencer = MattingInferencer(config, ckpt, device)
        elif self.type == 'inpainting':
            self.inferencer = InpaintingInferencer(config, ckpt, device)
        elif self.type == 'translation':
            self.inferencer = TranslationInferencer(config, ckpt, device)
        elif self.type == 'restoration':
            self.inferencer = RestorationInferencer(config, ckpt, device)
        elif self.type == 'video_restoration':
            self.inferencer = VideoRestorationInferencer(config, ckpt, device)
        elif self.type == 'video_interpolation':
            self.inferencer = VideoInterpolationInferencer(config, ckpt, device)
        else:
            raise ValueError(f'Unknown inferencer type: {self.type}')
    
    def __call__(self, **kwargs) -> Union[Dict, List[Dict]]:
        return self.inferencer(**kwargs)

