# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
from mmengine import FileClient
from mmengine.dataset import BaseDataset
from mmengine.logging import MMLogger

from mmedit.registry import DATASETS
from .data_utils import expanduser, find_folders, get_samples


@DATASETS.register_module()
class BasicConditionalDataset(BaseDataset):
    """Custom dataset for conditional GAN. This class is based on the
    combination of `BaseDataset` (https://github.com/open-
    mmlab/mmclassification/blob/1.x/mmcls/datasets/base_dataset.py)  # noqa and
    `CustomDataset` (https://github.com/open-
    mmlab/mmclassification/blob/1.x/mmcls/datasets/custom.py).  # noqa.

    The dataset supports two kinds of annotation format.

    1. A line-based annotation file (e.g., txt) is provided, and each line indicates a sample:

       The sample files: ::

           data_prefix/
           ├── folder_1
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           └── folder_2
               ├── 123.png
               ├── nsdf3.png
               └── ...

       The annotation file (the first column is the image path and the second
       column is the index of category): ::

            folder_1/xxx.png 0
            folder_1/xxy.png 1
            folder_2/123.png 5
            folder_2/nsdf3.png 3
            ...

       Please specify the name of categories by the argument ``classes``
       or ``metainfo``.

    2. A dict-based annotation file (e.g., json) is provided, key and value
       indicate the path and label of the sample:

       The sample files: ::

           data_prefix/
           ├── folder_1
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           └── folder_2
               ├── 123.png
               ├── nsdf3.png
               └── ...

       The annotation file (the key is the image path and the value column
       is the label): ::

            {
                "folder_1/xxx.png": [1, 2, 3, 4],
                "folder_1/xxy.png": [2, 4, 1, 0],
                "folder_2/123.png": [0, 9, 8, 1],
                "folder_2/nsdf3.png", [1, 0, 0, 2],
                ...
            }

       In this kind of annotation, labels can be any type and not restricted to an index.

    3. The samples are arranged in the specific way: ::

           data_prefix/
           ├── class_x
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           │       └── xxz.png
           └── class_y
               ├── 123.png
               ├── nsdf3.png
               ├── ...
               └── asd932_.png

    If the ``ann_file`` is specified, the dataset will be generated by the
    first two ways, otherwise, try the third way.

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for the data. Defaults to ''.
        extensions (Sequence[str]): A sequence of allowed extensions. Defaults
            to ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif').
        lazy_init (bool): Whether to load annotation during instantiation.
            In some cases, such as visualization, only the meta information of
            the dataset is needed, which is not necessary to load annotation
            file. ``Basedataset`` can skip load annotations to save time by set
            ``lazy_init=False``. Defaults to False.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 lazy_init: bool = False,
                 classes: Union[str, Sequence[str], None] = None,
                 **kwargs):
        assert (ann_file or data_prefix or data_root), \
            'One of `ann_file`, `data_root` and `data_prefix` must '\
            'be specified.'
        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))

        ann_file = expanduser(ann_file)
        metainfo = self._compat_classes(metainfo, classes)
        self.extensions = tuple(set([i.lower() for i in extensions]))

        super().__init__(
            # The base class requires string ann_file but this class doesn't
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            # Force to lazy_init for some modification before loading data.
            lazy_init=True,
            **kwargs)

        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

    def _find_samples(self, file_client):
        """find samples from ``data_prefix``."""
        classes, folder_to_idx = find_folders(self.img_prefix, file_client)
        samples, empty_classes = get_samples(
            self.img_prefix,
            folder_to_idx,
            is_valid_file=self.is_valid_file,
            file_client=file_client,
        )

        if len(samples) == 0:
            raise RuntimeError(
                f'Found 0 files in subfolders of: {self.data_prefix}. '
                f'Supported extensions are: {",".join(self.extensions)}')

        if self.CLASSES is not None:
            assert len(self.CLASSES) == len(classes), \
                f"The number of subfolders ({len(classes)}) doesn't match " \
                f'the number of specified classes ({len(self.CLASSES)}). ' \
                'Please check the data folder.'
        else:
            self._metainfo['classes'] = tuple(classes)

        if empty_classes:
            logger = MMLogger.get_current_instance()
            logger.warning(
                'Found no valid file in the folder '
                f'{", ".join(empty_classes)}. '
                f"Supported extensions are: {', '.join(self.extensions)}")

        self.folder_to_idx = folder_to_idx

        return samples

    def load_data_list(self):
        """Load image paths and gt_labels."""
        if self.img_prefix:
            file_client = FileClient.infer_client(uri=self.img_prefix)

        if not self.ann_file:
            samples = self._find_samples(file_client)
        elif self.ann_file.endswith('json'):
            samples = mmengine.fileio.io.load(self.ann_file)
            samples = [[name, label] for name, label in samples.items()]
        else:
            lines = mmengine.list_from_file(self.ann_file)
            samples = [x.strip().rsplit(' ', 1) for x in lines]

        def add_prefix(filename, prefix=''):
            if not prefix:
                return filename
            else:
                return file_client.join_path(prefix, filename)

        data_list = []
        for filename, gt_label in samples:
            img_path = add_prefix(filename, self.img_prefix)
            # convert digit label to int
            if isinstance(gt_label, str):
                gt_label = int(gt_label) if gt_label.isdigit() else gt_label
            info = {'img_path': img_path, 'gt_label': gt_label}
            data_list.append(info)
        return data_list

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)

    @property
    def img_prefix(self):
        """The prefix of images."""
        return self.data_prefix['img_path']

    @property
    def CLASSES(self):
        """Return all categories names."""
        return self._metainfo.get('classes', None)

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {cat: i for i, cat in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """

        gt_labels = np.array(
            [self.get_data_info(i)['gt_label'] for i in range(len(self))])
        return gt_labels

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image category of specified index.
        """

        return [int(self.get_data_info(idx)['gt_label'])]

    def _compat_classes(self, metainfo, classes):
        """Merge the old style ``classes`` arguments to ``metainfo``."""
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmengine.list_from_file(expanduser(classes))
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        elif classes is not None:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if metainfo is None:
            metainfo = {}

        if classes is not None:
            metainfo = {'classes': tuple(class_names), **metainfo}

        return metainfo

    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True."""
        super().full_init()

        #  To support the standard OpenMMLab 2.0 annotation format. Generate
        #  metainfo in internal format from standard metainfo format.
        if 'categories' in self._metainfo and 'classes' not in self._metainfo:
            categories = sorted(
                self._metainfo['categories'], key=lambda x: x['id'])
            self._metainfo['classes'] = tuple(
                [cat['category_name'] for cat in categories])

    def __repr__(self):
        """Print the basic information of the dataset.

        Returns:
            str: Formatted string.
        """
        head = 'Dataset ' + self.__class__.__name__
        body = []
        if self._fully_initialized:
            body.append(f'Number of samples: \t{self.__len__()}')
        else:
            body.append("Haven't been initialized")

        if self.CLASSES is not None:
            body.append(f'Number of categories: \t{len(self.CLASSES)}')
        else:
            body.append('The `CLASSES` meta info is not set.')

        body.extend(self.extra_repr())

        if len(self.pipeline.transforms) > 0:
            body.append('With transforms:')
            for t in self.pipeline.transforms:
                body.append(f'    {t}')

        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = []
        body.append(f'Annotation file: \t{self.ann_file}')
        body.append(f'Prefix of images: \t{self.img_prefix}')
        return body
