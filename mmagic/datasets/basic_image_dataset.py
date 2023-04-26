# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
from typing import Callable, List, Optional, Tuple, Union

from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend, list_from_file

from mmagic.registry import DATASETS

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class BasicImageDataset(BaseDataset):
    """BasicImageDataset for open source projects in OpenMMLab/MMagic.

    This dataset is designed for low-level vision tasks with image,
    such as super-resolution and inpainting.

    The annotation file is optional.

    If use annotation file, the annotation format can be shown as follows.

    .. code-block:: none

        Case 1 (CelebA-HQ):

            000001.png
            000002.png

        Case 2 (DIV2K):

            0001_s001.png (480,480,3)
            0001_s002.png (480,480,3)
            0001_s003.png (480,480,3)
            0002_s001.png (480,480,3)
            0002_s002.png (480,480,3)

        Case 3 (Vimeo90k):

            00001/0266 (256, 448, 3)
            00001/0268 (256, 448, 3)

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img=None, ann=None).
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        filename_tmpl (dict): Template for each filename. Note that the
            template excludes the file extension. Default: dict().
        search_key (str): The key used for searching the folder to get
            data_list. Default: 'gt'.
        backend_args (dict, optional): Arguments to instantiate the prefix of
            uri corresponding backend. Defaults to None.
        suffix (str or tuple[str], optional):  File suffix
            that we are interested in. Default: None.
        recursive (bool): If set to True, recursively scan the
            directory. Default: False.

    Note:

        Assume the file structure as the following:

        .. code-block:: none

            mmagic (root)
            ├── mmagic
            ├── tools
            ├── configs
            ├── data
            │   ├── DIV2K
            │   │   ├── DIV2K_train_HR
            │   │   │   ├── image.png
            │   │   ├── DIV2K_train_LR_bicubic
            │   │   │   ├── X2
            │   │   │   ├── X3
            │   │   │   ├── X4
            │   │   │   │   ├── image_x4.png
            │   │   ├── DIV2K_valid_HR
            │   │   ├── DIV2K_valid_LR_bicubic
            │   │   │   ├── X2
            │   │   │   ├── X3
            │   │   │   ├── X4
            │   ├── places
            │   │   ├── test_set
            │   │   ├── train_set
            |   |   ├── meta
            |   |   |    ├── Places365_train.txt
            |   |   |    ├── Places365_val.txt

    Examples:

        Case 1: Loading DIV2K dataset for training a SISR model.

        .. code-block:: python

            dataset = BasicImageDataset(
                ann_file='',
                metainfo=dict(
                    dataset_type='div2k',
                    task_name='sisr'),
                data_root='data/DIV2K',
                data_prefix=dict(
                    gt='DIV2K_train_HR', img='DIV2K_train_LR_bicubic/X4'),
                filename_tmpl=dict(img='{}_x4', gt='{}'),
                pipeline=[])

        Case 2: Loading places dataset for training an inpainting model.

        .. code-block:: python

            dataset = BasicImageDataset(
                ann_file='meta/Places365_train.txt',
                metainfo=dict(
                    dataset_type='places365',
                    task_name='inpainting'),
                data_root='data/places',
                data_prefix=dict(gt='train_set'),
                pipeline=[])
    """

    METAINFO = dict(dataset_type='basic_image_dataset', task_name='editing')

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 filename_tmpl: dict = dict(),
                 search_key: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 img_suffix: Optional[Union[str, Tuple[str]]] = IMG_EXTENSIONS,
                 recursive: bool = False,
                 **kwards):

        for key in data_prefix:
            if key not in filename_tmpl:
                filename_tmpl[key] = '{}'

        if search_key is None:
            keys = list(data_prefix.keys())
            search_key = keys[0]
        self.search_key = search_key
        self.filename_tmpl = filename_tmpl
        self.use_ann_file = (ann_file != '')
        if backend_args is None:
            self.backend_args = None
        else:
            self.backend_args = backend_args.copy()
        self.img_suffix = img_suffix
        self.recursive = recursive
        self.file_backend = get_file_backend(
            uri=data_root, backend_args=backend_args)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwards)

    def load_data_list(self) -> List[dict]:
        """Load data list from folder or annotation file.

        Returns:
            list[dict]: A list of annotation.
        """

        path_list = self._get_path_list()

        data_list = []
        for file in path_list:
            basename, ext = osp.splitext(file)
            if basename.startswith(os.sep):
                # Avoid absolute-path-like annotations
                basename = basename[1:]
            data = dict(key=basename)
            for key in self.data_prefix:
                path = osp.join(self.data_prefix[key],
                                (f'{self.filename_tmpl[key].format(basename)}'
                                 f'{ext}'))
                data[f'{key}_path'] = path
            data_list.append(data)

        return data_list

    def _get_path_list(self):
        """Get list of paths from annotation file or folder of dataset.

        Returns:
            list[dict]: A list of paths.
        """

        path_list = []
        if self.use_ann_file:
            path_list = self._get_path_list_from_ann()
        else:
            path_list = self._get_path_list_from_folder()

        return path_list

    def _get_path_list_from_ann(self):
        """Get list of paths from annotation file.

        Returns:
            List: List of paths.
        """

        ann_list = list_from_file(
            self.ann_file, backend_args=self.backend_args)
        path_list = []
        for ann in ann_list:
            if ann.isspace() or ann == '':
                continue
            path = ann.split(' ')[0]
            # Compatible with Windows file systems
            path = path.replace('/', os.sep)
            path_list.append(path)

        return path_list

    def _get_path_list_from_folder(self):
        """Get list of paths from folder.

        Returns:
            List: List of paths.
        """

        path_list = []
        folder = self.data_prefix[self.search_key]
        tmpl = self.filename_tmpl[self.search_key].format('')
        virtual_path = self.filename_tmpl[self.search_key].format('.*')
        for img_path in self.file_backend.list_dir_or_file(
                dir_path=folder,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=self.recursive,
        ):
            basename, ext = osp.splitext(img_path)
            if re.match(virtual_path, basename):
                img_path = img_path.replace(tmpl + ext, ext)
                path_list.append(img_path)

        return path_list
