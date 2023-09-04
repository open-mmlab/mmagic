# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

from mmagic.datasets import MSCoCoDataset


class TestMSCoCoDatasets:

    @classmethod
    def setup_class(cls):
        cls.data_root = Path(__file__).parent.parent / 'data' / 'coco'

    def test_mscoco(self):

        # test basic usage
        dataset = MSCoCoDataset(data_root=self.data_root, pipeline=[])
        assert dataset[0] == dict(
            gt_prompt='a good meal',
            img_path=os.path.join(self.data_root, 'train2014',
                                  'COCO_train2014_000000000009.jpg'),
            sample_idx=0)

        # test with different phase
        dataset = MSCoCoDataset(
            data_root=self.data_root, phase='val', pipeline=[])
        assert dataset[0] == dict(
            gt_prompt='a pair of slippers',
            img_path=os.path.join(self.data_root, 'val2014',
                                  'COCO_val2014_000000000042.jpg'),
            sample_idx=0)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
