# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmagic.registry import DATASETS

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class PairedImageDataset(BaseDataset):
    """General paired image folder dataset for image generation.

    It assumes that the training directory is '/path/to/data/train'.
    During test time, the directory is '/path/to/data/test'. '/path/to/data'
    can be initialized by args 'dataroot'. Each sample contains a pair of
    images concatenated in the w dimension (A|B).

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of paired images.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        test_dir (str): Subfolder of dataroot which contain test images.
            Default: 'test'.
    """

    def __init__(self,
                 data_root,
                 pipeline,
                 io_backend: Optional[str] = None,
                 test_mode=False,
                 test_dir='test'):
        phase = test_dir if test_mode else 'train'
        self.data_root = osp.join(str(data_root), phase)

        if io_backend is None:
            self.file_backend = get_file_backend(uri=data_root)
        else:
            self.file_backend = get_file_backend(
                backend_args={'backend': io_backend})

        super().__init__(
            data_root=self.data_root, pipeline=pipeline, test_mode=test_mode)
        # self.data_infos = self.load_annotations()

    def load_data_list(self):
        """Load paired image paths.

        Returns:
            list[dict]: List that contains paired image paths.
        """
        data_infos = []
        pair_paths = sorted(self.scan_folder(self.data_root))
        for pair_path in pair_paths:
            data_infos.append(dict(pair_path=pair_path))

        return data_infos

    def scan_folder(self, path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: Image list obtained from the given folder.
        """
        imgs_list = self.file_backend.list_dir_or_file(
            path, list_dir=False, suffix=IMG_EXTENSIONS, recursive=True)
        images = [self.file_backend.join_path(path, img) for img in imgs_list]
        assert images, f'{path} has no valid image file.'
        return images
