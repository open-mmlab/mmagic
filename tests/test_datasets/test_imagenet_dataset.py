# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase

from mmagic.datasets import ImageNet

DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../data/dataset/'))


class TestImageNetDataset(TestCase):

    def test_load_data_list(self):

        with self.assertRaisesRegex(
                AssertionError, r"\(3\) doesn't match .* classes \(1000\)"):
            ImageNet(data_root=DATA_DIR, data_prefix=DATA_DIR)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
