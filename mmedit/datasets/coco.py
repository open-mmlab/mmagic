# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset
from mmengine.fileio import load

from mmedit.registry import DATASETS


@DATASETS.register_module()
class CocoDataset(BaseDataset):
    """Dataset for COCO."""

    METAINFO = {
        'dataset_type': 'colorization_dataset',
        'task_name': 'colorization',
    }

    def load_data_list(self) -> List[dict]:

        annotations = load(self.ann_file)

        assert annotations, f'annotation file "{self.ann_file}" is empty.'

        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        data_list = []
        for raw_data_info in raw_data_list:
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                data_list.append(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Join data_root to each path in data_info."""

        data_info = raw_data_info.copy()
        for key in raw_data_info:
            if 'path' in key:
                data_info['gt_img_path'] = osp.join(self.data_root,
                                                    data_info[key])

        return data_info
