# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch

from mmedit.utils import ConfigType
from .base_mmedit_inferencer import BaseMMEditInferencer
from .conditional_inferencer import ConditionalInferencer
from .inpainting_inferencer import InpaintingInferencer
from .matting_inferencer import MattingInferencer
from .restoration_inferencer import RestorationInferencer
from .translation_inferencer import TranslationInferencer
from .unconditional_inferencer import UnconditionalInferencer
from .video_interpolation_inferencer import VideoInterpolationInferencer
from .video_restoration_inferencer import VideoRestorationInferencer


class MMEditInferencer(BaseMMEditInferencer):
    """Class to assign task to different inferencers.

    Args:
        type (str): Inferencer type.
        config (str or ConfigType): Model config or the path to it.
        ckpt (str, optional): Path to the checkpoint.
        device (str, optional): Device to run inference. If None, the best
            device will be automatically used.
    """

    def __init__(
        self,
        type: Optional[str] = None,
        config: Optional[Union[ConfigType, str]] = None,
        ckpt: Optional[str] = None,
        device: torch.device = None,
        extra_parameters: Optional[Dict] = None,
    ) -> None:
        self.type = type
        if self.type == 'conditional':
            self.inferencer = ConditionalInferencer(config, ckpt, device,
                                                    extra_parameters)
        elif self.type == 'unconditional':
            self.inferencer = UnconditionalInferencer(config, ckpt, device,
                                                      extra_parameters)
        elif self.type == 'matting':
            self.inferencer = MattingInferencer(config, ckpt, device,
                                                extra_parameters)
        elif self.type == 'inpainting':
            self.inferencer = InpaintingInferencer(config, ckpt, device,
                                                   extra_parameters)
        elif self.type == 'translation':
            self.inferencer = TranslationInferencer(config, ckpt, device,
                                                    extra_parameters)
        elif self.type == 'restoration':
            self.inferencer = RestorationInferencer(config, ckpt, device,
                                                    extra_parameters)
        elif self.type == 'video_restoration':
            self.inferencer = VideoRestorationInferencer(
                config, ckpt, device, extra_parameters)
        elif self.type == 'video_interpolation':
            self.inferencer = VideoInterpolationInferencer(
                config, ckpt, device, extra_parameters)
        else:
            raise ValueError(f'Unknown inferencer type: {self.type}')

    def __call__(self, **kwargs) -> Union[Dict, List[Dict]]:
        """Call the inferencer.

        Args:
            kwargs: Keyword arguments for the inferencer.

        Returns:
            Union[Dict, List[Dict]]: Results of inference pipeline.
        """
        return self.inferencer(**kwargs)

    def get_extra_parameters(self) -> List[str]:
        """Each inferencer may has its own parameters. Call this function to
        get these parameters.

        Returns:
            List[str]: List of unique parameters.
        """
        return self.inferencer.get_extra_parameters()
