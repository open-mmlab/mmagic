# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
from typing import List, Union

from mmengine.dataset import BaseDataset
from mmengine.fileio import load

from mmagic.registry import DATASETS


@DATASETS.register_module()
class AdobeComp1kDataset(BaseDataset):
    """Adobe composition-1k dataset.

    The dataset loads (alpha, fg, bg) data and apply specified transforms to
    the data. You could specify whether composite merged image online or load
    composited merged image in pipeline.

    Example for online comp-1k dataset:

    ::

        [
            {
                "alpha": 'alpha/000.png',
                "fg": 'fg/000.png',
                "bg": 'bg/000.png'
            },
            {
                "alpha": 'alpha/001.png',
                "fg": 'fg/001.png',
                "bg": 'bg/001.png'
            },
        ]

    Example for offline comp-1k dataset:

    ::

        [
            {
                "alpha": 'alpha/000.png',
                "merged": 'merged/000.png',
                "fg": 'fg/000.png',
                "bg": 'bg/000.png'
            },
            {
                "alpha": 'alpha/001.png',
                "merged": 'merged/001.png',
                "fg": 'fg/001.png',
                "bg": 'bg/001.png'
            },
        ]

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        **kwargs: Other arguments passed to
            :class:`mmengine.dataset.BaseDataset`.

    Examples:
        See unit-tests
        TODO: Move some codes in unittest here
    """

    # TODO: Support parsing folder structures without annotation files.

    METAINFO = dict(dataset_type='matting_dataset', task_name='matting')

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        In order to be compatible to both new and old annotation format,
        we copy implementations from mmengine and do some modifications.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = load(self.ann_file)
        assert annotations, f'annotation file "{self.ann_file}" is empty.'
        if isinstance(annotations, list):
            # Old annotation format, we get data_list only
            raw_data_list = annotations
        elif isinstance(annotations, dict):
            # New annotation format, follow original routine in base class
            if 'data_list' not in annotations or 'metainfo' not in annotations:
                raise ValueError('Annotation must have data_list and metainfo '
                                 'keys')
            metainfo = annotations['metainfo']
            raw_data_list = annotations['data_list']

            # Meta information load from annotation file will not influence the
            # existed meta information load from `BaseDataset.METAINFO` and
            # `metainfo` arguments defined in constructor.
            for k, v in metainfo.items():
                self._metainfo.setdefault(k, v)
        else:
            raise TypeError(
                f'The annotations loaded from annotation file '
                f'should be a list or dict, but got {type(annotations)}!')

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_list must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Join data_root to each path in data_info."""

        data_info = raw_data_info.copy()
        for key in raw_data_info:
            data_info[key] = osp.join(self.data_root, data_info[key])

        return data_info
