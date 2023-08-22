# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_interpolator import BasicInterpolator


@MODELS.register_module()
class FLAVR(BasicInterpolator):
    """Basic model for video interpolation.

    It must contain a generator that takes frames as inputs and outputs an
    interpolated frame. It also has a pixel-wise loss for training.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(
            generator=generator,
            pixel_loss=pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            required_frames=4,
            step_frames=1,
            pretrained=pretrained)

    @staticmethod
    def merge_frames(input_tensors, output_tensors):
        """merge input frames and output frames.

        Interpolate a frame between the given two frames.

        Merged from
            [[in1, in2, in3, in4], [in2, in3, in4, in5], ...]
            [[out1], [out2], [out3], ...]
        to
            [in1, in2, out1, in3, out2, ..., in(-3), out(-1), in(-2), in(-1)]

        Args:
            input_tensors (Tensor): The input frames with shape [n, 4, c, h, w]
            output_tensors (Tensor): The output frames with shape
                [n, 1, c, h, w].

        Returns:
            list[np.array]: The final frames.
        """

        num_frames = input_tensors.shape[0]
        result = [tensor2img(input_tensors[0, 0])]
        for i in range(num_frames):
            result.append(tensor2img(input_tensors[i, 1]))
            result.append(tensor2img(output_tensors[i, 0]))
        result.append(tensor2img(input_tensors[-1, 2]))
        result.append(tensor2img(input_tensors[-1, 3]))

        return result
