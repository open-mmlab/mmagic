# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel

from mmagic.registry import MODELS
from mmagic.structures import DataSample


@MODELS.register_module()
class BaseEditModel(BaseModel):
    """Base model for image and video editing.

    It must contain a generator that takes frames as inputs and outputs an
    interpolated frame. It also has a pixel-wise loss for training.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.

    Attributes:
        init_cfg (dict, optional): Initialization config dict.
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`. Default: None.
    """

    def __init__(self,
                 generator: dict,
                 pixel_loss: dict,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = MODELS.build(generator)

        # loss
        self.pixel_loss = MODELS.build(pixel_loss)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                **kwargs) -> Union[torch.Tensor, List[DataSample], dict]:
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
        if isinstance(inputs, dict):
            inputs = inputs['img']
        if mode == 'tensor':
            return self.forward_tensor(inputs, data_samples, **kwargs)

        elif mode == 'predict':
            predictions = self.forward_inference(inputs, data_samples,
                                                 **kwargs)
            predictions = self.convert_to_datasample(predictions, data_samples,
                                                     inputs)
            return predictions

        elif mode == 'loss':
            return self.forward_train(inputs, data_samples, **kwargs)

    def convert_to_datasample(self, predictions: DataSample,
                              data_samples: DataSample,
                              inputs: Optional[torch.Tensor]
                              ) -> List[DataSample]:
        """Add predictions and destructed inputs (if passed) to data samples.

        Args:
            predictions (DataSample): The predictions of the model.
            data_samples (DataSample): The data samples loaded from
                dataloader.
            inputs (Optional[torch.Tensor]): The input of model. Defaults to
                None.

        Returns:
            List[DataSample]: Modified data samples.
        """

        if inputs is not None:
            destructed_input = self.data_preprocessor.destruct(
                inputs, data_samples, 'img')
            data_samples.set_tensor_data({'input': destructed_input})
        # split to list of data samples
        data_samples = data_samples.split()
        predictions = predictions.split()

        for data_sample, pred in zip(data_samples, predictions):
            data_sample.output = pred

        return data_samples

    def forward_tensor(self,
                       inputs: torch.Tensor,
                       data_samples: Optional[List[DataSample]] = None,
                       **kwargs) -> torch.Tensor:
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: result of simple forward.
        """

        feats = self.generator(inputs, **kwargs)

        return feats

    def forward_inference(self,
                          inputs: torch.Tensor,
                          data_samples: Optional[List[DataSample]] = None,
                          **kwargs) -> DataSample:
        """Forward inference. Returns predictions of validation, testing, and
        simple inference.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            DataSample: predictions.
        """

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        feats = self.data_preprocessor.destruct(feats, data_samples)

        # create a stacked data sample here
        predictions = DataSample(pred_img=feats.cpu())

        return predictions

    def forward_train(self,
                      inputs: torch.Tensor,
                      data_samples: Optional[List[DataSample]] = None,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """Forward training. Returns dict of losses of training.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: Dict of losses.
        """

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        batch_gt_data = data_samples.gt_img

        loss = self.pixel_loss(feats, batch_gt_data)

        return dict(loss=loss)
