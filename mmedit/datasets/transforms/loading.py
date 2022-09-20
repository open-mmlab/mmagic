# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Optional, Tuple

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient, list_from_file

from mmedit.registry import TRANSFORMS
from mmedit.utils import (bbox2mask, brush_stroke_mask, get_irregular_mask,
                          random_bbox)


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load a single image or image frames from corresponding paths. Required
    Keys:
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
            Defaults to None.
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
        channel_order: str = 'bgr',
        imdecode_backend: Optional[str] = None,
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
            self.file_client_args = None
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
        results[f'{self.key}_channel_order'] = self.channel_order
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

        if (self.file_client_args is not None) and (self.file_client_args.get(
                'backend', None) == 'lmdb'):
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


@TRANSFORMS.register_module()
class LoadMask(BaseTransform):
    """Load Mask for multiple types.

    For different types of mask, users need to provide the corresponding
    config dict.

    Example config for bbox:

    .. code-block:: python

        config = dict(img_shape=(256, 256), max_bbox_shape=128)

    Example config for irregular:

    .. code-block:: python

        config = dict(
            img_shape=(256, 256),
            num_vertices=(4, 12),
            max_angle=4.,
            length_range=(10, 100),
            brush_width=(10, 40),
            area_ratio_range=(0.15, 0.5))

    Example config for ff:

    .. code-block:: python

        config = dict(
            img_shape=(256, 256),
            num_vertices=(4, 12),
            mean_angle=1.2,
            angle_range=0.4,
            brush_width=(12, 40))

    Example config for set:

    .. code-block:: python

        config = dict(
            mask_list_file='xxx/xxx/ooxx.txt',
            prefix='/xxx/xxx/ooxx/',
            io_backend='disk',
            color_type='unchanged',
            file_client_kwargs=dict()
        )

        The mask_list_file contains the list of mask file name like this:
            test1.jpeg
            test2.jpeg
            ...
            ...

        The prefix gives the data path.

    Args:
        mask_mode (str): Mask mode in ['bbox', 'irregular', 'ff', 'set',
            'file']. Default: 'bbox'.
            * bbox: square bounding box masks.
            * irregular: irregular holes.
            * ff: free-form holes from DeepFillv2.
            * set: randomly get a mask from a mask set.
            * file: get mask from 'mask_path' in results.
        mask_config (dict): Params for creating masks. Each type of mask needs
            different configs. Default: None.
    """

    def __init__(self, mask_mode='bbox', mask_config=None):
        self.mask_mode = mask_mode
        self.mask_config = dict() if mask_config is None else mask_config
        assert isinstance(self.mask_config, dict)

        # set init info if needed in some modes
        self._init_info()

    def _init_info(self):
        if self.mask_mode == 'set':
            # get mask list information
            self.io_backend = self.mask_config['io_backend']
            self.color_type = self.mask_config['color_type']
            self.file_prefix = self.mask_config['prefix']
            self.file_client_kwargs = self.mask_config['file_client_kwargs']
            self.file_client = None

            mask_list_file = self.mask_config['mask_list_file']
            self.mask_list = list_from_file(
                mask_list_file, file_client_args=self.file_client_kwargs)
            self.mask_list = [
                osp.join(self.file_prefix, i) for i in self.mask_list
            ]
            self.mask_set_size = len(self.mask_list)
        elif self.mask_mode == 'file':
            self.io_backend = 'disk'
            self.color_type = 'unchanged'
            self.file_client_kwargs = dict()
            self.file_client = None

    def _get_random_mask_from_set(self):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend)
        # minus 1 to avoid out of range error
        mask_idx = np.random.randint(0, self.mask_set_size)
        mask_bytes = self.file_client.get(self.mask_list[mask_idx])
        mask = mmcv.imfrombytes(mask_bytes, flag=self.color_type)  # HWC, BGR
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)
        else:
            mask = mask[:, :, 0:1]

        mask[mask > 0] = 1.
        return mask

    def _get_mask_from_file(self, path):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend,
                                          **self.file_client_kwargs)
        mask_bytes = self.file_client.get(path)
        mask = mmcv.imfrombytes(mask_bytes, flag=self.color_type)  # HWC, BGR
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)
        else:
            mask = mask[:, :, 0:1]

        mask[mask > 0] = 1.
        return mask

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        if self.mask_mode == 'bbox':
            mask_bbox = random_bbox(**self.mask_config)
            mask = bbox2mask(self.mask_config['img_shape'], mask_bbox)
            results['mask_bbox'] = mask_bbox
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(**self.mask_config)
        elif self.mask_mode == 'set':
            mask = self._get_random_mask_from_set()
        elif self.mask_mode == 'ff':
            mask = brush_stroke_mask(**self.mask_config)
        elif self.mask_mode == 'file':
            mask = self._get_mask_from_file(results['mask_path'])
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        results['mask'] = mask
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(mask_mode='{self.mask_mode}')"


@TRANSFORMS.register_module()
class GetSpatialDiscountMask(BaseTransform):
    """Get spatial discounting mask constant.

    Spatial discounting mask is first introduced in:
    Generative Image Inpainting with Contextual Attention.

    Args:
        gamma (float, optional): Gamma for computing spatial discounting.
            Defaults to 0.99.
        beta (float, optional): Beta for computing spatial discounting.
            Defaults to 1.5.
    """

    def __init__(self, gamma=0.99, beta=1.5):
        self.gamma = gamma
        self.beta = beta

    def spatial_discount_mask(self, mask_width, mask_height):
        """Generate spatial discounting mask constant.

        Args:
            mask_width (int): The width of bbox hole.
            mask_height (int): The height of bbox height.

        Returns:
            np.ndarray: Spatial discounting mask.
        """
        w, h = np.meshgrid(np.arange(mask_width), np.arange(mask_height))
        grid_stack = np.stack([h, w], axis=2)
        mask_values = (self.gamma**(np.minimum(
            grid_stack, [mask_height - 1, mask_width - 1] - grid_stack) *
                                    self.beta)).max(
                                        axis=2, keepdims=True)

        return mask_values

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        mask_bbox = results['mask_bbox']
        mask = results['mask']
        mask_height, mask_width = mask_bbox[-2:]
        discount_hole = self.spatial_discount_mask(mask_width, mask_height)
        discount_mask = np.zeros_like(mask)
        discount_mask[mask_bbox[0]:mask_bbox[0] + mask_height,
                      mask_bbox[1]:mask_bbox[1] + mask_width,
                      ...] = discount_hole

        results['discount_mask'] = discount_mask

        return results

    def __repr__(self):
        return self.__class__.__name__ + (f'(gamma={self.gamma}, '
                                          f'beta={self.beta})')


@TRANSFORMS.register_module()
class LoadPairedImageFromFile(LoadImageFromFile):
    """Load a pair of images from file.

    Each sample contains a pair of images, which are concatenated in the w
    dimension (a|b). This is a special loading class for generation paired
    dataset. It loads a pair of images as the common loader does and crops
    it into two images with the same shape in different domains.

    Required key is "pair_path". Added or modified keys are "pair",
    "pair_ori_shape", "ori_pair", "img_{domain_a}", "img_{domain_b}",
    "img_{domain_a}_path", "img_{domain_b}_path", "img_{domain_a}_ori_shape",
    "img_{domain_b}_ori_shape", "ori_img_{domain_a}" and
    "ori_img_{domain_b}".

    Args:
        key (str): Keys in results to find corresponding path.
        domain_a (str, Optional): One of the paired image domain. Defaults
            to 'A'.
        domain_b (str, Optional): The other of the paired image domain.
            Defaults to 'B'.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            candidates are 'cv2', 'turbojpeg', 'pillow', and 'tifffile'.
            Defaults to None.
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
        io_backend (str, optional): io backend where images are store. Defaults
            to None.
    """

    def __init__(self,
                 key: str,
                 domain_a: str = 'A',
                 domain_b: str = 'B',
                 color_type: str = 'color',
                 channel_order: str = 'bgr',
                 imdecode_backend: Optional[str] = None,
                 use_cache: bool = False,
                 to_float32: bool = False,
                 to_y_channel: bool = False,
                 save_original_img: bool = False,
                 file_client_args: Optional[dict] = None):
        super().__init__(key, color_type, channel_order, imdecode_backend,
                         use_cache, to_float32, to_y_channel,
                         save_original_img, file_client_args)
        assert isinstance(domain_a, str)
        assert isinstance(domain_b, str)
        self.domain_a = domain_a
        self.domain_b = domain_b

    def transform(self, results: dict) -> dict:
        """Functions to load paired images.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filename = results[f'{self.key}_path']

        image = self._load_image(filename)
        image = self._convert(image)
        if self.save_original_img:
            ori_image = image.copy()
        shape = image.shape

        # crop pair into a and b
        w = shape[1]
        if w % 2 != 0:
            raise ValueError(
                f'The width of image pair must be even number, but got {w}.')
        new_w = w // 2
        image_a = image[:, :new_w, :]
        image_b = image[:, new_w:, :]

        results[f'img_{self.domain_a}'] = image_a
        results[f'img_{self.domain_b}'] = image_b
        results[f'img_{self.domain_a}_path'] = filename
        results[f'img_{self.domain_b}_path'] = filename
        results[f'img_{self.domain_a}_ori_shape'] = image_a.shape
        results[f'img_{self.domain_b}_ori_shape'] = image_b.shape
        if self.save_original_img:
            results[f'ori_img_{self.domain_a}'] = image_a.copy()
            results[f'ori_img_{self.domain_b}'] = image_b.copy()

        results[self.key] = image
        results[f'ori_{self.key}_shape'] = shape
        results[f'{self.key}_channel_order'] = self.channel_order
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_image

        return results
