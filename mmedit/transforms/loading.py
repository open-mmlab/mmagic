# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Optional, Tuple

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient

from ..registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load a single image or image frames from corresponding paths.

    Required Keys:

    - [Key]_path

    New Keys:

    - [KEY]
    - ori_[KEY]_shape
    - ori_[KEY]

    Args:
        key (str): Keys in results to find corresponding path.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            candidates are 'cv2', 'turbojpeg', 'pillow', and 'tifffile'.
            Defaults to 'cv2'.
        use_cache (bool): If True, load all images at once. Default: False.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        to_y_channel (bool): Whether to convert the loaded image to y channel.
            Only support 'rgb2ycbcr' and 'rgb2ycbcr'
            Defaults to False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            If not specified, will infer from file uri.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``None``.
    """

    def __init__(
        self,
        key: str,
        color_type: str = 'color',
        channel_order='bgr',
        imdecode_backend: str = 'cv2',
        use_cache: bool = False,
        to_float32: bool = False,
        to_y_channel: bool = False,
        save_original_img: bool = False,
        file_client_args: Optional[dict] = None,
    ) -> None:

        self.key = key
        self.color_type = color_type
        self.channel_order = channel_order
        self.imdecode_backend = imdecode_backend
        self.save_original_img = save_original_img

        if file_client_args is None:
            # lasy init at loading
            self.file_client_args = dict()
            self.file_client = None
        else:
            self.file_client_args = file_client_args.copy()
            self.file_client = FileClient(**self.file_client_args)

        # cache
        self.use_cache = use_cache
        self.cache = dict()

        # convert
        self.to_float32 = to_float32
        self.to_y_channel = to_y_channel

    def transform(self, results: dict) -> dict:
        """Functions to load image or frames.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filenames = results[f'{self.key}_path']

        if not isinstance(filenames, (List, Tuple)):
            filenames = [str(filenames)]
            is_frames = False
        else:
            filenames = [str(v) for v in filenames]
            is_frames = True

        images = []
        shapes = []
        if self.save_original_img:
            ori_imgs = []

        for filename in filenames:
            img = self._load_image(filename)
            img = self._convert(img)
            images.append(img)
            shapes.append(img.shape)
            if self.save_original_img:
                ori_imgs.append(img.copy())

        if not is_frames:
            images = images[0]
            shapes = shapes[0]
            if self.save_original_img:
                ori_imgs = ori_imgs[0]

        results[self.key] = images
        results[f'ori_{self.key}_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        return results

    def _load_image(self, filename):
        """Load an image from file.

        Args:
            filename (str): Path of image file.

        Returns:
            np.ndarray: Image.
        """
        if self.file_client is None:
            self.file_client = FileClient.infer_client(
                uri=filename, file_client_args=self.file_client_args)

        if self.file_client_args.get('backend', None) == 'lmdb':
            filename, _ = osp.splitext(osp.basename(filename))

        if filename in self.cache:
            img_bytes = self.cache[filename]
        else:
            img_bytes = self.file_client.get(filename)
            if self.use_cache:
                self.cache[filename] = img_bytes

        img = mmcv.imfrombytes(
            content=img_bytes,
            flag=self.color_type,
            channel_order=self.channel_order,
            backend=self.imdecode_backend)

        return img

    def _convert(self, img: np.ndarray):
        """Convert an image to the require format.

        Args:
            img (np.ndarray): The original image.

        Returns:
            np.ndarray: The converted image.
        """

        if self.to_y_channel:

            if self.channel_order.lower() == 'rgb':
                img = mmcv.rgb2ycbcr(img, y_only=True)
            elif self.channel_order.lower() == 'bgr':
                img = mmcv.bgr2ycbcr(img, y_only=True)
            else:
                raise ValueError('Currently support only "bgr2ycbcr" or '
                                 '"bgr2ycbcr".')

            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

        if self.to_float32:
            img = img.astype(np.float32)

        return img

    def __repr__(self):

        repr_str = (f'{self.__class__.__name__}('
                    f'key={self.key}, '
                    f'color_type={self.color_type}, '
                    f'channel_order={self.channel_order}, '
                    f'imdecode_backend={self.imdecode_backend}, '
                    f'use_cache={self.use_cache}, '
                    f'to_float32={self.to_float32}, '
                    f'to_y_channel={self.to_y_channel}, '
                    f'save_original_img={self.save_original_img}, '
                    f'file_client_args={self.file_client_args})')

        return repr_str
