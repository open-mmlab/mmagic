# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapperDict

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from .color_utils import get_colorization_data, lab2rgb


@MODELS.register_module()
class InstColorization(BaseModel):
    """Colorization InstColorization  method.

    This Colorization is implemented according to the paper:
        Instance-aware Image Colorization, CVPR 2020

    Adapted from 'https://github.com/ericsujw/InstColorization.git'
    'InstColorization/models/train_model'
    Copyright (c) 2020, Su, under MIT License.

    Args:
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        image_model (dict): Config for single image model
        instance_model (dict): Config for instance model
        fusion_model (dict): Config for fusion model
        color_data_opt (dict): Option for colorspace conversion
        which_direction (str): AtoB or BtoA
        loss (dict): Config for loss.
        init_cfg  (str): Initialization config dict. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self,
                 data_preprocessor: Union[dict, Config],
                 image_model,
                 instance_model,
                 fusion_model,
                 color_data_opt,
                 which_direction='AtoB',
                 loss=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        # colorization networks
        # image_model: used to colorize a single image
        self.image_model = MODELS.build(image_model)

        # instance model: used to colorize cropped instance
        self.instance_model = MODELS.build(instance_model)

        # fusion model: input a single image with related instance features
        self.fusion_model = MODELS.build(fusion_model)

        self.color_data_opt = color_data_opt
        self.which_direction = which_direction

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                **kwargs):
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``inputs`` and ``data_samples`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.val_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
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
            ForwardResults:

                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of
                  :obj:`BaseDataElement` for computing metric
                  and getting inference result.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict`` or tensor for custom use.
        """

        if mode == 'tensor':
            return self.forward_tensor(inputs, data_samples, **kwargs)

        elif mode == 'predict':
            predictions = self.forward_inference(inputs, data_samples,
                                                 **kwargs)
            predictions = self.convert_to_datasample(data_samples, predictions)
            return predictions

        elif mode == 'loss':
            return self.forward_train(inputs, data_samples, **kwargs)

    def convert_to_datasample(self, inputs, data_samples):
        """Add predictions and destructed inputs (if passed) to data samples.

        Args:
            inputs (Optional[torch.Tensor]): The input of model. Defaults to
                None.
            data_samples (List[DataSample]): The data samples loaded from
                dataloader.

        Returns:
            List[DataSample]: Modified data samples.
        """

        for data_sample, output in zip(inputs, data_samples):
            data_sample.output = output
        return inputs

    def forward_train(self, inputs, data_samples=None, **kwargs):
        """Forward function for training."""
        raise NotImplementedError(
            'Instance Colorization has not supported training.')

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapperDict) -> Dict[str, torch.Tensor]:
        """Train step function.

        Args:
            data (List[dict]): Batch of data as input.
            optim_wrapper (dict[torch.optim.Optimizer]): Dict with optimizers
                for generator and discriminator (if have).
        Returns:
            dict: Dict with loss, information for logger, the number of
                samples and results for visualization.
        """
        raise NotImplementedError(
            'Instance Colorization has not supported training.')

    def forward_inference(self, inputs, data_samples=None, **kwargs):
        """Forward inference. Returns predictions of validation, testing.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            List[DataSample]: predictions.
        """
        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        feats = self.data_preprocessor.destruct(feats, data_samples)
        predictions = []
        for idx in range(feats.shape[0]):
            pred_img = feats[idx].to('cpu')
            predictions.append(
                DataSample(
                    pred_img=pred_img, metainfo=data_samples[idx].metainfo))

        return predictions

    def forward_tensor(self, inputs, data_samples):
        """Forward function in tensor mode.

        Args:
            inputs (torch.Tensor): Input tensor.
            data_sample (dict): Dict contains data sample.

        Returns:
            dict: Dict contains output results.
        """

        #  prepare data

        assert len(data_samples) == 1, \
            'fusion model supports only one image due to different numbers '\
            'of instances of different images'

        full_img_data = get_colorization_data(inputs, self.color_data_opt)
        AtoB = self.which_direction == 'AtoB'

        # preprocess input for a single image
        full_real_A = full_img_data['A' if AtoB else 'B']
        full_hint_B = full_img_data['hint_B']
        full_mask_B = full_img_data['mask_B']

        if not data_samples.empty_box[0]:
            # preprocess instance input
            cropped_img = data_samples.cropped_img[0]
            box_info_list = [
                data_samples.box_info[0], data_samples.box_info_2x[0],
                data_samples.box_info_4x[0], data_samples.box_info_8x[0]
            ]
            cropped_data = get_colorization_data(cropped_img,
                                                 self.color_data_opt)

            real_A = cropped_data['A' if AtoB else 'B']
            hint_B = cropped_data['hint_B']
            mask_B = cropped_data['mask_B']

            # network forward
            _, output, feature_map = self.instance_model(
                real_A, hint_B, mask_B)
            output = self.fusion_model(full_real_A, full_hint_B, full_mask_B,
                                       feature_map, box_info_list)

        else:
            _, output, _ = self.image_model(full_real_A, full_hint_B,
                                            full_mask_B)

        output = [
            full_real_A.type(torch.cuda.FloatTensor),
            output.type(torch.cuda.FloatTensor)
        ]
        output = torch.cat(output, dim=1)
        output = torch.clamp(lab2rgb(output, self.color_data_opt), 0.0, 1.0)
        return output
