# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
from pathlib import Path

from mmedit.registry import DATASETS
from mmedit.datasets import CocoDataset


# todo 完成coco的单元测试编写
class TestCOCOStuff:
    DATASET_TYPE = 'CocoDataset'

    ann_file = 'test.json'
    data_root = "../.."

    DEFAULT_ARGS = dict(
        data_root=data_root + '/train2017',
        data_prefix=dict(gt='data_large'),
        ann_file=ann_file,
        pipeline=[],
        test_mode=False
    )

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        dataset = dataset_class(**self.DEFAULT_ARGS)

        assert dataset.mateinfo == {
            'dataset_type': 'colorization_dataset',
            'task_name': 'colorization',
        }

        # 对拿到的数据列表和数据进行判断
        