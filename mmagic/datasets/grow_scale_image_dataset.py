# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmengine import print_log
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmagic.registry import DATASETS


@DATASETS.register_module()
class GrowScaleImgDataset(BaseDataset):
    """Grow Scale Unconditional Image Dataset.

    This dataset is similar with ``UnconditionalImageDataset``, but offer
    more dynamic functionalities for the supporting complex algorithms, like
    PGGAN.

    Highlight functionalities:

    #. Support growing scale dataset. The motivation is to decrease data
       pre-processing load in CPU. In this dataset, you can provide
       ``imgs_roots`` like:

        .. code-block:: python

            {'64': 'path_to_64x64_imgs',
             '512': 'path_to_512x512_imgs'}

       Then, in training scales lower than 64x64, this dataset will set
       ``self.imgs_root`` as 'path_to_64x64_imgs';
    #. Offer ``samples_per_gpu`` according to different scales. In this
       dataset, ``self.samples_per_gpu`` will help runner to know the updated
       batch size.

    Basically, This dataset contains raw images for training unconditional
    GANs. Given a root dir, we will recursively find all images in this root.
    The transformation on data is defined by the pipeline.

    Args:
        imgs_root (str): Root path for unconditional images.
        pipeline (list[dict | callable]): A sequence of data transforms.
        len_per_stage (int, optional): The length of dataset for each scale.
            This args change the length dataset by concatenating or extracting
            subset. If given a value less than 0., the original length will be
            kept. Defaults to 1e6.
        gpu_samples_per_scale (dict | None, optional): Dict contains
            ``samples_per_gpu`` for each scale. For example, ``{'32': 4}`` will
            set the scale of 32 with ``samples_per_gpu=4``, despite other scale
            with ``samples_per_gpu=self.gpu_samples_base``.
        gpu_samples_base (int, optional): Set default ``samples_per_gpu`` for
            each scale. Defaults to 32.
        io_backend (str, optional): The storage backend type. Options are
            "disk", "ceph", "memcached", "lmdb", "http" and "petrel".
            Default: None.
        test_mode (bool, optional): If True, the dataset will work in test
            mode. Otherwise, in train mode. Default to False.
    """

    _VALID_IMG_SUFFIX = ('.jpg', '.png', '.jpeg', '.JPEG')

    def __init__(self,
                 data_roots: dict,
                 pipeline,
                 len_per_stage=int(1e6),
                 gpu_samples_per_scale=None,
                 gpu_samples_base=32,
                 io_backend: Optional[str] = None,
                 file_lists: Optional[Union[str, dict]] = None,
                 test_mode=False):

        assert isinstance(data_roots, dict)
        self.data_roots = data_roots
        self._img_scales = sorted([int(x) for x in data_roots.keys()])
        self._curr_scale = self._img_scales[0]
        self._actual_curr_scale = self._curr_scale
        self.data_root = self.data_roots[str(self._curr_scale)]

        # len_per_stage = -1, keep the original length
        self.len_per_stage = len_per_stage
        self.curr_stage = 0
        self.gpu_samples_per_scale = gpu_samples_per_scale
        if self.gpu_samples_per_scale is not None:
            assert isinstance(self.gpu_samples_per_scale, dict)
        else:
            self.gpu_samples_per_scale = dict()
        self.gpu_samples_base = gpu_samples_base

        if io_backend is None:
            data_root_ = list(data_roots.values())[0]
            self.file_backend = get_file_backend(uri=data_root_)
        else:
            self.file_backend = get_file_backend(
                backend_args={'backend': io_backend})

        # use current data root to initialize and do not support
        # `serialize_data`
        super().__init__(
            data_root=self.data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=False)

        # print basic dataset information to check the validity
        print_log(repr(self), 'current')

    def load_data_list(self):
        """Load annotations."""
        # recursively find all of the valid images from imgs_root
        data_list = self.file_backend.list_dir_or_file(
            self.data_root,
            list_dir=False,
            suffix=self._VALID_IMG_SUFFIX,
            recursive=True)
        self.data_list = [
            self.file_backend.join_path(self.data_root, x) for x in data_list
        ]

        if self.len_per_stage > 0:
            self.concat_imgs_list_to(self.len_per_stage)
        self.samples_per_gpu = self.gpu_samples_per_scale.get(
            str(self._actual_curr_scale), self.gpu_samples_base)
        return self.data_list

    def update_annotations(self, curr_scale):
        """Update annotations.

        Args:
            curr_scale (int): Current image scale.

        Returns:
            bool: Whether to update.
        """
        if curr_scale == self._actual_curr_scale:
            return False

        for scale in self._img_scales:
            if curr_scale <= scale:
                self._curr_scale = scale
                break
            if scale == self._img_scales[-1]:
                assert RuntimeError(
                    f'Cannot find a suitable scale for {curr_scale}')
        self._actual_curr_scale = curr_scale
        self.data_root = self.data_roots[str(self._curr_scale)]
        self.load_data_list()
        # print basic dataset information to check the validity
        print_log('Update Dataset: ' + repr(self), 'current')
        return True

    def concat_imgs_list_to(self, num):
        """Concat image list to specified length.

        Args:
            num (int): The length of the concatenated image list.
        """

        if num <= len(self.data_list):
            self.data_list = self.data_list[:num]
            return

        concat_factor = (num // len(self.data_list)) + 1
        imgs = self.data_list * concat_factor
        self.data_list = imgs[:num]

    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        results = dict(gt_path=self.data_list[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        results = dict(gt_path=self.data_list[idx])
        return self.pipeline(results)

    def __getitem__(self, idx):
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetch is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        if not self.test_mode:
            return self.prepare_train_data(idx)

        return self.prepare_test_data(idx)

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        dataset_name = self.__class__
        imgs_root = self.data_root
        num_imgs = len(self)
        return (f'dataset_name: {dataset_name}, total {num_imgs} images in '
                f'imgs_root: {imgs_root}')
