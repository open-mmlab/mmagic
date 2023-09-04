# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import pickle
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from mmagic.registry import DATASETS
from mmagic.utils import register_all_modules

DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../data/dataset/'))

register_all_modules()


class TestCIFAR10(TestCase):
    DATASET_TYPE = 'CIFAR10'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        data_prefix = tmpdir.name
        cls.DEFAULT_ARGS = dict(
            data_prefix=data_prefix, pipeline=[], test_mode=False)

        dataset_class = DATASETS.get(cls.DATASET_TYPE)
        base_folder = osp.join(data_prefix, dataset_class.base_folder)
        os.mkdir(base_folder)

        cls.fake_imgs = np.random.randint(
            0, 255, size=(6, 3 * 32 * 32), dtype=np.uint8)
        cls.fake_labels = np.random.randint(0, 10, size=(6, ))
        cls.fake_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        batch1 = dict(
            data=cls.fake_imgs[:2], labels=cls.fake_labels[:2].tolist())
        with open(osp.join(base_folder, 'data_batch_1'), 'wb') as f:
            f.write(pickle.dumps(batch1))

        batch2 = dict(
            data=cls.fake_imgs[2:4], labels=cls.fake_labels[2:4].tolist())
        with open(osp.join(base_folder, 'data_batch_2'), 'wb') as f:
            f.write(pickle.dumps(batch2))

        test_batch = dict(
            data=cls.fake_imgs[4:], fine_labels=cls.fake_labels[4:].tolist())
        with open(osp.join(base_folder, 'test_batch'), 'wb') as f:
            f.write(pickle.dumps(test_batch))

        meta = {dataset_class.meta['key']: cls.fake_classes}
        meta_filename = dataset_class.meta['filename']
        with open(osp.join(base_folder, meta_filename), 'wb') as f:
            f.write(pickle.dumps(meta))

        dataset_class.train_list = [['data_batch_1', None],
                                    ['data_batch_2', None]]
        dataset_class.test_list = [['test_batch', None]]
        dataset_class.meta['md5'] = None

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test overriding metainfo by `metainfo` argument
        cfg = {**self.DEFAULT_ARGS, 'metainfo': {'classes': ('bus', 'car')}}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))

        # Test overriding metainfo by `classes` argument
        cfg = {**self.DEFAULT_ARGS, 'classes': ['bus', 'car']}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))

        classes_file = osp.join(DATA_DIR, 'classes.txt')
        cfg = {**self.DEFAULT_ARGS, 'classes': classes_file}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))
        self.assertEqual(dataset.class_to_idx, {'bus': 0, 'car': 1})

        # Test invalid classes
        cfg = {**self.DEFAULT_ARGS, 'classes': dict(classes=1)}
        with self.assertRaisesRegex(ValueError, "type <class 'dict'>"):
            dataset_class(**cfg)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.CLASSES, dataset_class.METAINFO['classes'])

        data_info = dataset[0]
        fake_img = self.fake_imgs[0].reshape(3, 32, 32).transpose(1, 2, 0)
        np.testing.assert_equal(data_info['gt'], fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_labels[0])
        assert data_info['gt_channel_order'] == 'RGB'

        # Test with test_mode=True
        cfg = {**self.DEFAULT_ARGS, 'test_mode': True}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        fake_img = self.fake_imgs[4].reshape(3, 32, 32).transpose(1, 2, 0)
        np.testing.assert_equal(data_info['gt'], fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_labels[4])
        assert data_info['gt_channel_order'] == 'RGB'

        # Test load meta
        cfg = {**self.DEFAULT_ARGS, 'lazy_init': True}
        dataset = dataset_class(**cfg)
        dataset._metainfo = {}
        dataset.full_init()
        self.assertEqual(dataset.CLASSES, self.fake_classes)

        cfg = {**self.DEFAULT_ARGS, 'lazy_init': True}
        dataset = dataset_class(**cfg)
        dataset._metainfo = {}
        dataset.meta['filename'] = 'invalid'
        with self.assertRaisesRegex(RuntimeError, 'not found or corrupted'):
            dataset.full_init()

        # Test automatically download
        with patch(
                'mmagic.datasets.cifar10_dataset.download_and_extract_archive'
        ) as mock:
            cfg = {**self.DEFAULT_ARGS, 'lazy_init': True, 'test_mode': True}
            dataset = dataset_class(**cfg)
            dataset.test_list = [['invalid_batch', None]]
            with self.assertRaisesRegex(AssertionError, 'Download failed'):
                dataset.full_init()
            mock.assert_called_once_with(
                dataset.url,
                dataset.data_prefix['root'],
                filename=dataset.filename,
                md5=dataset.tgz_md5)

        with self.assertRaisesRegex(RuntimeError, '`download=True`'):
            cfg = {
                **self.DEFAULT_ARGS, 'lazy_init': True,
                'test_mode': True,
                'download': False
            }
            dataset = dataset_class(**cfg)
            dataset.test_list = [['test_batch', 'invalid_md5']]
            dataset.full_init()

        # Test different backend
        cfg = {
            **self.DEFAULT_ARGS, 'lazy_init': True,
            'data_prefix': 'http://openmmlab/cifar'
        }
        dataset = dataset_class(**cfg)
        dataset._check_integrity = MagicMock(return_value=False)
        with self.assertRaisesRegex(RuntimeError, 'http://openmmlab/cifar'):
            dataset.full_init()

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS, 'lazy_init': True}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Prefix of data: \t{dataset.data_prefix["root"]}',
                      repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
