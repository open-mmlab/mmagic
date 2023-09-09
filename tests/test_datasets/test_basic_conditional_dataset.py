# Copyright (c) OpenMMLab. All rights reserved.
# import os
import os.path as osp
from unittest import TestCase

import numpy as np

from mmagic.datasets import BasicConditionalDataset
from mmagic.utils import register_all_modules

register_all_modules()

DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../data/dataset/'))


class TestBasicConditonalDataset(TestCase):

    def test_init(self):
        ann_file = osp.abspath(osp.join(DATA_DIR, 'anno.txt'))
        dataset = BasicConditionalDataset(
            data_root=DATA_DIR,
            ann_file=ann_file,
            metainfo={'classes': ('bus', 'car')})

        self.assertEqual(dataset.CLASSES, ('bus', 'car'))
        self.assertFalse(dataset.test_mode)
        self.assertNotIn('With transforms:', repr(dataset))

        classes_file = osp.abspath(osp.join(DATA_DIR, 'classes.txt'))
        dataset = BasicConditionalDataset(
            data_root=DATA_DIR, ann_file=ann_file, classes=classes_file)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))
        self.assertEqual(dataset.class_to_idx, {'bus': 0, 'car': 1})

        ann_file = osp.abspath(osp.join(DATA_DIR, 'wrong.yml'))
        with self.assertRaises(TypeError):
            BasicConditionalDataset(data_root=DATA_DIR, ann_file=ann_file)

        gt_labels = dataset.get_gt_labels()
        print(type(gt_labels))
        self.assertTrue((gt_labels == np.array([0, 1, 1])).all())

        for idx, tar_ids in enumerate([0, 1, 1]):
            cat_ids = dataset.get_cat_ids(idx)
            self.assertEqual(cat_ids, [tar_ids])

        data = dataset[0]
        self.assertEqual(data['sample_idx'], 0)
        self.assertEqual(data['gt_label'], 0)
        self.assertIn('a/1.JPG', data['gt_path'])

        # test data prefix --> b/subb
        dataset = BasicConditionalDataset(data_root=DATA_DIR, data_prefix='b')
        self.assertIn('subb', dataset.CLASSES)

        dataset = BasicConditionalDataset(
            data_root=DATA_DIR, data_prefix={'gt_path': 'b'})
        self.assertIn('subb', dataset.CLASSES)

        # test runtime error --> no samples
        with self.assertRaises(RuntimeError):
            dataset = BasicConditionalDataset(data_root=osp.dirname(__file__))

        # test assert error --> class list is not same
        with self.assertRaises(AssertionError):
            dataset = BasicConditionalDataset(
                data_root=DATA_DIR, classes=['1', '2', '3', '4'])

        # test Value error --> wrong classes type input
        with self.assertRaises(ValueError):
            dataset = BasicConditionalDataset(
                data_root=DATA_DIR, classes=dict(a=1))

        # test raise warning --> find empty classes -->
        # TODO: how to get logger's output?
        dataset = BasicConditionalDataset(data_root=DATA_DIR)

        # test lazy init
        dataset = BasicConditionalDataset(
            data_root=DATA_DIR,
            lazy_init=True,
            pipeline=[dict(type='PackInputs')])
        self.assertFalse(dataset._fully_initialized)
        self.assertIn("Haven't been initialized", repr(dataset))
        self.assertIn('With transforms:', repr(dataset))

        # test load label from json file
        ann_file = osp.abspath(osp.join(DATA_DIR, 'anno.json'))
        dataset = BasicConditionalDataset(
            data_root=DATA_DIR,
            ann_file=ann_file,
            lazy_init=True,
            pipeline=[dict(type='PackInputs')])
        self.assertEqual(dataset[0]['data_samples'].gt_label.label.tolist(),
                         [1, 2, 3, 4])
        self.assertEqual(dataset[1]['data_samples'].gt_label.label.tolist(),
                         [1, 4, 5, 3])


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
