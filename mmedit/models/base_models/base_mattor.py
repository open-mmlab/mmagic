# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel

from mmedit.registry import MODELS
from mmedit.structures import EditDataSample, PixelData

DataSamples = Optional[Union[list, torch.Tensor]]
ForwardResults = Union[Dict[str, torch.Tensor], List[EditDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


def _pad(batch_image: torch.Tensor,
         ds_factor: int,
         mode: str = 'reflect') -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad image to a multiple of give down-sampling factor."""

    h, w = batch_image.shape[-2:]  # NCHW

    new_h = ds_factor * ((h - 1) // ds_factor + 1)
    new_w = ds_factor * ((w - 1) // ds_factor + 1)

    pad_h = new_h - h
    pad_w = new_w - w
    pad = (pad_h, pad_w)
    if new_h != h or new_w != w:
        pad_width = (0, pad_w, 0, pad_h)  # torch.pad in reverse order
        batch_image = F.pad(batch_image, pad_width, mode)

    return batch_image, pad


def _interpolate(batch_image: torch.Tensor,
                 ds_factor: int,
                 mode: str = 'bicubic'
                 ) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Resize image to multiple of give down-sampling factor."""

    h, w = batch_image.shape[-2:]  # NCHW

    new_h = h - (h % ds_factor)
    new_w = w - (w % ds_factor)

    size = (new_h, new_w)
    if new_h != h or new_w != w:
        batch_image = F.interpolate(batch_image, size=size, mode=mode)

    return batch_image, size


class BaseMattor(BaseModel, metaclass=ABCMeta):
    """Base class for trimap-based matting models.

    A matting model must contain a backbone which produces `pred_alpha`,
    a dense prediction with the same height and width of input image.
    In some cases (such as DIM), the model has a refiner which refines
    the prediction of the backbone.

    Subclasses should overwrite the following functions:

    - :meth:`_forward_train`, to return a loss
    - :meth:`_forward_test`, to return a prediction
    - :meth:`_forward`, to return raw tensors

    For test, this base class provides functions to resize inputs and
    post-process pred_alphas to get predictions

    Args:
        backbone (dict): Config of backbone.
        data_preprocessor (dict): Config of data_preprocessor.
            See :class:`MattorPreprocessor` for details.
        init_cfg (dict, optional): Initialization config dict.
        train_cfg (dict): Config of training.
            Customized by subclassesCustomized bu In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing.
            In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
    """

    def __init__(self,
                 data_preprocessor: Union[dict, Config],
                 backbone: dict,
                 init_cfg: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None):
        # Build data_preprocessor in BaseModel
        # Initialize weights in BaseModule
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.train_cfg = ConfigDict(
            train_cfg) if train_cfg is not None else ConfigDict()
        self.test_cfg = ConfigDict(
            test_cfg) if test_cfg is not None else ConfigDict()

        self.backbone = MODELS.build(backbone)

    def resize_inputs(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        """Pad or interpolate images and trimaps to multiple of given
        factor."""

        resize_method = self.test_cfg['resize_method']
        resize_mode = self.test_cfg['resize_mode']
        size_divisor = self.test_cfg['size_divisor']

        batch_images = batch_inputs[:, :3, :, :]
        batch_trimaps = batch_inputs[:, 3:, :, :]

        if resize_method == 'pad':
            batch_images, _ = _pad(batch_images, size_divisor, resize_mode)
            batch_trimaps, _ = _pad(batch_trimaps, size_divisor, resize_mode)
        elif resize_method == 'interp':
            batch_images, _ = _interpolate(batch_images, size_divisor,
                                           resize_mode)
            batch_trimaps, _ = _interpolate(batch_trimaps, size_divisor,
                                            'nearest')
        else:
            raise NotImplementedError

        return torch.cat((batch_images, batch_trimaps), dim=1)

    def restore_size(self, pred_alpha: torch.Tensor,
                     data_sample: EditDataSample) -> torch.Tensor:
        """Restore the predicted alpha to the original shape.

        The shape of the predicted alpha may not be the same as the shape of
        original input image. This function restores the shape of the predicted
        alpha.

        Args:
            pred_alpha (torch.Tensor): A single predicted alpha of
                shape (1, H, W).
            data_sample (EditDataSample): Data sample containing
                original shape as meta data.

        Returns:
            torch.Tensor: The reshaped predicted alpha.
        """
        resize_method = self.test_cfg['resize_method']
        resize_mode = self.test_cfg['resize_mode']

        ori_h, ori_w = data_sample.ori_merged_shape[:2]
        if resize_method == 'pad':
            pred_alpha = pred_alpha[:, :ori_h, :ori_w]
        elif resize_method == 'interp':
            pred_alpha = F.interpolate(
                pred_alpha.unsqueeze(0), size=(ori_h, ori_w), mode=resize_mode)
            pred_alpha = pred_alpha[0]  # 1,H,W

        return pred_alpha

    def postprocess(
        self,
        batch_pred_alpha: torch.Tensor,  # N, 1, H, W, float32
        data_samples: List[EditDataSample],
    ) -> List[EditDataSample]:
        """Post-process alpha predictions.

        This function contains the following steps:
            1. Restore padding or interpolation
            2. Mask alpha prediction with trimap
            3. Clamp alpha prediction to 0-1
            4. Convert alpha prediction to uint8
            5. Pack alpha prediction into EditDataSample

        Currently only batch_size 1 is actually supported.

        Args:
            batch_pred_alpha (torch.Tensor): A batch of predicted alpha
                of shape (N, 1, H, W).
            data_samples (List[EditDataSample]): List of data samples.

        Returns:
            List[EditDataSample]: A list of predictions.
                Each data sample contains a pred_alpha,
                which is a torch.Tensor with dtype=uint8, device=cuda:0
        """

        assert batch_pred_alpha.ndim == 4  # N, 1, H, W, float32
        assert len(batch_pred_alpha) == len(data_samples) == 1

        predictions = []
        for pa, ds in zip(batch_pred_alpha, data_samples):
            pa = self.restore_size(pa, ds)  # 1, H, W
            pa = pa[0]  # H, W

            pa.clamp_(min=0, max=1)
            ori_trimap = ds.ori_trimap[:, :, 0]  # H, W
            pa[ori_trimap == 255] = 1
            pa[ori_trimap == 0] = 0

            pa *= 255
            pa.round_()
            pa = pa.to(dtype=torch.uint8)
            # pa = pa.cpu().numpy()
            pa_sample = EditDataSample(pred_alpha=PixelData(data=pa))
            predictions.append(pa_sample)

        return predictions

    def forward(self,
                inputs: torch.Tensor,
                data_samples: DataSamples = None,
                mode: str = 'tensor') -> List[EditDataSample]:
        """General forward function.

        Args:
            inputs (torch.Tensor): A batch of inputs.
                with image and trimap concatenated alone channel dimension.
            data_samples (List[EditDataSample], optional):
                A list of data samples, containing:
                - Ground-truth alpha / foreground / background to compute loss
                - other meta information
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``. Default: 'tensor'.

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            List[EditDataElement]:
                Sequence of predictions packed into EditDataElement
        """
        if mode == 'tensor':
            raw = self._forward(inputs)
            return raw
        elif mode == 'predict':
            # Pre-process runs in runner
            inputs = self.resize_inputs(inputs)
            batch_pred_alpha = self._forward_test(inputs)
            predictions = self.postprocess(batch_pred_alpha, data_samples)
            predictions = self.convert_to_datasample(data_samples, predictions)
            return predictions
        elif mode == 'loss':
            loss = self._forward_train(inputs, data_samples)
            return loss
        else:
            raise ValueError('Invalid forward mode.')

    def convert_to_datasample(self, inputs: DataSamples,
                              data_samples: List[EditDataSample]
                              ) -> List[EditDataSample]:
        for data_sample, output in zip(inputs, data_samples):
            data_sample.output = output
        return inputs
