# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapperDict

from mmedit.models.utils import get_colorization_data, lab2rgb
from mmedit.registry import MODULES
from mmedit.structures import EditDataSample, PixelData


@MODULES.register_module()
class InstColorization(BaseModel):

    def __init__(self,
                 data_preprocessor: Union[dict, Config],
                 full_model,
                 instance_model,
                 stage,
                 ngf,
                 output_nc,
                 avg_loss_alpha,
                 ab_norm,
                 ab_max,
                 ab_quant,
                 l_norm,
                 l_cent,
                 sample_Ps,
                 mask_cent,
                 which_direction='AtoB',
                 loss=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        # colorization networks
        # Stage 1 & 3. fusion model intergrates the image model
        self.full_model = MODULES.build(full_model)

        # Stage 2. instance model used for training instance colorization
        self.instance_model = MODULES.build(instance_model)

        self.stage = stage

        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        # self.ngf = ngf
        # self.output_nc = output_nc
        # self.avg_loss_alpha = avg_loss_alpha
        # self.mask_cent = mask_cent
        # self.which_direction = which_direction

        # self.encode_ab_opt = dict(
        #     ab_norm=ab_norm, ab_max=ab_max, ab_quant=ab_quant)
        # self.colorization_data_opt = dict(
        #     ab_thresh=0,
        #     ab_norm=ab_norm,
        #     l_norm=l_norm,
        #     l_cent=l_cent,
        #     sample_PS=sample_Ps,
        #     mask_cent=mask_cent,
        # )
        # self.lab2rgb_opt = dict(
        # ab_norm=ab_norm, l_norm=l_norm, l_cent=l_cent)
        # self.convert_params = dict(
        #     ab_thresh=0,
        #     ab_norm=ab_norm,
        #     l_norm=l_norm,
        #     l_cent=l_cent,
        #     sample_PS=sample_Ps,
        #     mask_cent=mask_cent,
        # )

        # # loss
        # self.loss_names = ['G', 'L1']
        # self.criterionL1 = self.loss
        # self.avg_losses = OrderedDict()
        # self.error_cnt = 0
        # for loss_name in self.loss_names:
        #     self.avg_losses[loss_name] = 0

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[EditDataSample]] = None,
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

    def forward_train(self, inputs, data_samples=None, **kwargs):
        raise NotImplementedError(
            'Instance Colorization has not supported training.')

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapperDict) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            'Instance Colorization has not supported training.')

    def forward_inference(self, inputs, data_samples=None, **kwargs):
        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        predictions = []
        for idx in range(feats.shape[0]):
            batch_tensor = feats[idx] * 127.5 + 127.5
            pred_img = PixelData(data=batch_tensor.to('cpu'))
            predictions.append(
                EditDataSample(
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
        print(data_samples)
        for dp in data_samples:
            print(dp.keys())

        data = data_samples[0]
        full_img = data.full_gray

        if not data.empty_box:
            cropped_img = data.cropped_gray
            box_info = data.box_info
            box_info_2x = data.box_info_2x
            box_info_4x = data.box_info_4x
            box_info_8x = data.box_info_8x
            cropped_data = get_colorization_data(cropped_img,
                                                 **self.convert_params)
            full_img_data = get_colorization_data(full_img,
                                                  **self.convert_params)
            self.set_input(cropped_data)
            self.set_fusion_input(
                full_img_data,
                [box_info, box_info_2x, box_info_4x, box_info_8x])
        else:
            full_img_data = get_colorization_data(full_img, ab_thresh=0)
            self.set_forward_without_box(full_img_data)

        self.fake_B_reg = self.generator(self.real_A, self.hint_B, self.mask_B,
                                         self.full_real_A, self.full_hint_B,
                                         self.full_mask_B, self.box_info_list)

        out_img = torch.clamp(
            lab2rgb(
                torch.cat((self.full_real_A.type(torch.cuda.FloatTensor),
                           self.fake_B_reg.type(torch.cuda.FloatTensor)),
                          dim=1), **self.lab2rgb_opt), 0.0, 1.0)

        return out_img

    def convert_to_datasample(self, inputs, data_samples):
        for data_sample, output in zip(inputs, data_samples):
            data_sample.output = output
        return inputs
