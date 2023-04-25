# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

from mmagic.registry import MODELS
from mmagic.utils import tensor2img
from .base_edit_model import BaseEditModel

# TODO tensor2img will be move


@MODELS.register_module()
class BasicInterpolator(BaseEditModel):
    """Basic model for video interpolation.

    It must contain a generator that takes frames as inputs and outputs an
    interpolated frame. It also has a pixel-wise loss for training.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        required_frames (int): Required frames in each process. Default: 2
        step_frames (int): Step size of video frame interpolation. Default: 1
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.

    Attributes:
        init_cfg (dict, optional): Initialization config dict.
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`.
    """

    def __init__(self,
                 generator: dict,
                 pixel_loss: dict,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 required_frames: int = 2,
                 step_frames: int = 1,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):

        super().__init__(
            generator=generator,
            pixel_loss=pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        # Required frames in each process
        self.required_frames = required_frames
        # Step size of video frame interpolation
        self.step_frames = step_frames

    def split_frames(self, input_tensors: torch.Tensor) -> torch.Tensor:
        """split input tensors for inference.

        Args:
            input_tensors (Tensor): Tensor of input frames with shape
                [1, t, c, h, w]

        Returns:
            Tensor: Split tensor with shape [t-1, 2, c, h, w]
        """

        num_frames = input_tensors.shape[1]

        result = [
            input_tensors[:, i:i + self.required_frames]
            for i in range(0, num_frames - self.required_frames +
                           1, self.step_frames)
        ]
        result = torch.cat(result, dim=0)

        return result

    @staticmethod
    def merge_frames(input_tensors: torch.Tensor,
                     output_tensors: torch.Tensor) -> list:
        """merge input frames and output frames.

        Interpolate a frame between the given two frames.

        Merged from
            [[in1, in2], [in2, in3], [in3, in4], ...]
            [[out1], [out2], [out3], ...]
        to
            [in1, out1, in2, out2, in3, out3, in4, ...]

        Args:
            input_tensors (Tensor): The input frames with shape [n, 2, c, h, w]
            output_tensors (Tensor): The output frames with shape
                [n, 1, c, h, w].

        Returns:
            list[np.array]: The final frames.
        """

        num_frames = input_tensors.shape[0]
        result = []
        for i in range(num_frames):
            result.append(tensor2img(input_tensors[i, 0]))
            result.append(tensor2img(output_tensors[i, 0]))
        result.append(tensor2img(input_tensors[-1, 1]))

        return result
