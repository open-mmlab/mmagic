# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.models.base_models import BasicInterpolator
from mmedit.registry import MODELS


@MODELS.register_module()
class CAIN(BasicInterpolator):
    """CAIN model for Video Interpolation.

    Paper: Channel Attention Is All You Need for Video Frame Interpolation
    Ref repo: https://github.com/myungsub/CAIN

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

    def forward_inference(self, inputs, data_samples=None):
        """Forward inference. Returns predictions of validation, testing, and
        simple inference.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            List[EditDataSample]: predictions.
        """

        predictions = super().forward_inference(
            inputs, data_samples, padding_flag=True)

        return predictions
