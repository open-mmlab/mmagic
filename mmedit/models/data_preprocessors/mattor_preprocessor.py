# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
# import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from mmedit.registry import MODELS
from mmedit.structures import EditDataSample, PixelData

DataSamples = Optional[Union[list, torch.Tensor]]
ForwardResults = Union[Dict[str, torch.Tensor], List[EditDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module()
class MattorPreprocessor(BaseDataPreprocessor):
    """DataPreprocessor for matting models.

    See base class ``BaseDataPreprocessor`` for detailed information.

    Workflow as follow :

    - Collate and move data to the target device.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Stack inputs to batch_inputs.

    Args:
        mean (Sequence[float or int]): The pixel mean of R, G, B channels.
            Defaults to [123.675, 116.28, 103.53].
        std (Sequence[float or int]): The pixel standard deviation of R, G, B
            channels. [58.395, 57.12, 57.375].
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to True.
        proc_inputs (str): Methods to process inputs. Default: 'normalize'.
            Available options are ``normalize``.
        proc_trimap (str): Methods to process gt tensors.
            Default: 'rescale_to_zero_one'.
            Available options are ``rescale_to_zero_one`` and ``as-is``.
        proc_gt (str): Methods to process gt tensors.
            Default: 'rescale_to_zero_one'.
            Available options are ``rescale_to_zero_one`` and ``ignore``.
    """

    def __init__(self,
                 mean: float = [123.675, 116.28, 103.53],
                 std: float = [58.395, 57.12, 57.375],
                 bgr_to_rgb: bool = True,
                 proc_inputs: str = 'normalize',
                 proc_trimap: str = 'rescale_to_zero_one',
                 proc_gt: str = 'rescale_to_zero_one'):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        self.bgr_to_rgb = bgr_to_rgb
        self.proc_inputs = proc_inputs
        self.proc_trimap = proc_trimap
        self.proc_gt = proc_gt

    def _proc_inputs(self, inputs: List[torch.Tensor]):
        if self.proc_inputs == 'normalize':
            # bgr to rgb
            if self.bgr_to_rgb and inputs[0].size(0) == 3:
                inputs = [_input[[2, 1, 0], ...] for _input in inputs]
            # Normalization.
            inputs = [(_input - self.mean) / self.std for _input in inputs]
            # Stack Tensor.
            batch_inputs = torch.stack(inputs)
        else:
            raise ValueError(
                f'proc_inputs = {self.proc_inputs} is not supported.')

        assert batch_inputs.ndim == 4
        return batch_inputs

    def _proc_trimap(self, trimaps: List[torch.Tensor]):
        batch_trimaps = torch.stack(trimaps)

        if self.proc_trimap == 'rescale_to_zero_one':
            batch_trimaps = batch_trimaps / 255.0  # uint8->float32
        elif self.proc_trimap == 'as_is':
            batch_trimaps = batch_trimaps.to(torch.float32)
        else:
            raise ValueError(
                f'proc_trimap = {self.proc_trimap} is not supported.')

        return batch_trimaps

    def _proc_gt(self, data_samples, key):
        assert key.startswith('gt')
        # Rescale gt_fg / gt_bg / gt_merged / gt_alpha to 0 to 1
        if self.proc_gt == 'rescale_to_zero_one':
            for ds in data_samples:
                try:
                    value = getattr(ds, key)
                except AttributeError:
                    continue

                ispixeldata = isinstance(value, PixelData)
                if ispixeldata:
                    value = value.data

                # !! DO NOT process trimap here, as trimap may have dim == 3
                if self.bgr_to_rgb and value[0].size(0) == 3:
                    value = value[[2, 1, 0], ...]

                value = value / 255.0  # uint8 -> float32 No inplace here

                assert value.ndim == 3

                if ispixeldata:
                    value = PixelData(data=value)
                setattr(ds, key, value)
        elif self.proc_gt == 'ignore':
            pass
        else:
            raise ValueError(f'proc_gt = {self.proc_gt} is not supported.')

        return data_samples

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, list]:
        """Pre-process input images, trimaps, ground-truth as configured.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Default: False.

        Returns:
            Tuple[torch.Tensor, list]:
                Batched inputs and list of data samples.
        """
        if not training:
            # Image may of different size when testing
            assert len(
                data['data_samples']) == 1, ('only batch_size=1 '
                                             'is supported for testing.')
        data = super().forward(data, training=training)

        images, trimaps, batch_data_samples = self.collate_data(data)

        batch_images = self._proc_inputs(images)
        batch_trimaps = self._proc_trimap(trimaps)

        if training:
            self._proc_gt(batch_data_samples, 'gt_fg')
            self._proc_gt(batch_data_samples, 'gt_bg')
            self._proc_gt(batch_data_samples, 'gt_merged')
            # For training, gt_alpha ranges from 0-1, is used to compute loss
            # For testing, ori_alpha is used
            self._proc_gt(batch_data_samples, 'gt_alpha')

        # Stack image and trimap along channel dimension
        # All existing models do concat at the start of forwarding
        # and data_sample is a very complex data structure
        # so this is a simple work-around to make codes simpler
        # print(f"batch_trimap.dtype = {batch_trimap.dtype}")

        assert batch_images.ndim == batch_trimaps.ndim == 4
        assert batch_images.shape[-2:] == batch_trimaps.shape[-2:], f"""
            Expect merged.shape[-2:] == trimap.shape[-2:],
            but got {batch_images.shape[-2:]} vs {batch_trimaps.shape[-2:]}
            """

        # N, (4/6), H, W
        batch_inputs = torch.cat((batch_images, batch_trimaps), dim=1)

        data['inputs'] = batch_inputs
        data['data_samples'] = batch_data_samples
        # return batch_inputs, batch_data_samples
        return data

    def collate_data(self, data: Sequence[dict]) -> Tuple[list, list, list]:
        """Collating and moving data to the target device.

        See base class ``BaseDataPreprocessor`` for detailed information.
        """
        inputs = [data_ for data_ in data['inputs']]
        trimaps = [data_.trimap.data for data_ in data['data_samples']]
        batch_data_samples = [data_ for data_ in data['data_samples']]

        # Move data from CPU to corresponding device.
        inputs = [_input.to(self.device) for _input in inputs]
        trimaps = [_trimap.to(self.device) for _trimap in trimaps]
        batch_data_samples = [
            data_sample.to(self.device) for data_sample in batch_data_samples
        ]
        return inputs, trimaps, batch_data_samples
