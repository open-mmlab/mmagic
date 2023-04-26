# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.models.base_models import BasicInterpolator
from mmagic.registry import MODELS
from mmagic.utils import tensor2img

# TODO tensor2img will be move


@MODELS.register_module()
class FLAVR(BasicInterpolator):
    """FLAVR model for video interpolation.

    Paper:
        FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation

    Ref repo: https://github.com/tarun005/FLAVR

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
