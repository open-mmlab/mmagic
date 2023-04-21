# Copyright (c) OpenMMLab. All rights reserved.
import logging
import re
from typing import Sequence

import numpy as np
import torch
from mmengine.visualization import Visualizer

from mmagic.registry import VISUALIZERS
from mmagic.structures import DataSample
from mmagic.utils import print_colored_log


@VISUALIZERS.register_module()
class ConcatImageVisualizer(Visualizer):
    """Visualize multiple images by concatenation.

    This visualizer will horizontally concatenate images belongs to different
    keys and vertically concatenate images belongs to different frames to
    visualize.

    Image to be visualized can be:
        - torch.Tensor or np.array
        - Image sequences of shape (T, C, H, W)
        - Multi-channel image of shape (1/3, H, W)
        - Single-channel image of shape (C, H, W)

    Args:
        fn_key (str): key used to determine file name for saving image.
            Usually it is the path of some input image. If the value is
            `dir/basename.ext`, the name used for saving will be basename.
        img_keys (str): keys, values of which are images to visualize.
        pixel_range (dict): min and max pixel value used to denormalize images,
            note that only float array or tensor will be denormalized,
            uint8 arrays are assumed to be unnormalized.
        bgr2rgb (bool): whether to convert the image from BGR to RGB.
        name (str): name of visualizer. Default: 'visualizer'.
        *args and \**kwargs: Other arguments are passed to `Visualizer`. # noqa
    """

    def __init__(self,
                 fn_key: str,
                 img_keys: Sequence[str],
                 pixel_range={},
                 bgr2rgb=False,
                 name: str = 'visualizer',
                 *args,
                 **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        self.fn_key = fn_key
        self.img_keys = img_keys
        self.pixel_range = pixel_range
        self.bgr2rgb = bgr2rgb

    def add_datasample(self, data_sample: DataSample, step=0) -> None:
        """Concatenate image and draw.

        Args:
            input (torch.Tensor): Single input tensor from data_batch.
            data_sample (DataSample): Single data_sample from data_batch.
            output (DataSample): Single prediction output by model.
            step (int): Global step value to record. Default: 0.
        """
        # Note:
        # with LocalVisBackend and default arguments, we have:
        # self.save_dir == runner._log_dir / 'vis_data'

        merged_dict = {
            **data_sample.to_dict(),
        }

        if 'output' in merged_dict.keys():
            merged_dict.update(**merged_dict['output'])

        fn = merged_dict[self.fn_key]
        if isinstance(fn, list):
            fn = fn[0]
        fn = re.split(r' |/|\\', fn)[-1]
        fn = fn.split('.')[0]

        img_list = []
        for k in self.img_keys:
            if k not in merged_dict:
                print_colored_log(
                    f'Key "{k}" not in data_sample or outputs',
                    level=logging.WARN)
                continue

            img = merged_dict[k]

            # PixelData
            if isinstance(img, dict) and ('data' in img):
                img = img['data']

            # Tensor to array
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
                if img.ndim == 3:
                    img = img.transpose(1, 2, 0)
                elif img.ndim == 4:
                    img = img.transpose(0, 2, 3, 1)

            # concat frame vertically
            if img.ndim == 4:
                img = np.concatenate(img, axis=0)

            # gray to 3 channel
            if (img.ndim == 3 and img.shape[2] == 1):
                img = np.concatenate((img, img, img), axis=2)

            # gray to 3 channel
            if img.ndim == 2:
                img = np.stack((img, img, img), axis=2)

            if self.bgr2rgb:
                img = img[..., ::-1]

            if img.dtype != np.uint8:
                # We assume uint8 type are not normalized
                if k in self.pixel_range:
                    min_, max_ = self.pixel_range.get(k)
                    img = ((img - min_) / (max_ - min_)) * 255
                img = img.clip(0, 255).round().astype(np.uint8)

            img_list.append(img)

        max_height = max(img.shape[0] for img in img_list)

        for i, img in enumerate(img_list):
            if img.shape[0] < max_height:
                img_list[i] = np.concatenate([
                    img,
                    np.ones((max_height - img.shape[0], *img.shape[1:]),
                            dtype=img.dtype) * 127
                ],
                                             axis=0)

        img_cat = np.concatenate(img_list, axis=1)

        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(fn, img_cat, step)
