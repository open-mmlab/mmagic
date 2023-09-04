# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

from torch.utils.data import Dataset

from mmagic.datasets.transforms import LoadImageFromFile
from mmagic.evaluation.functional.inception_utils import (
    get_inception_feat_cache_name_and_args, get_vgg_feat_cache_name_and_args)


def test_inception_feat_cache_name_args():
    dataloader = MagicMock()
    dataloader.dataset = MagicMock(spec=Dataset)
    dataloader.dataset.data_root = 'test_root'
    dataloader.dataset.data_prefix = 'test_prefix'
    dataloader.dataset.metainfo = dict(meta_info='test_meta')
    dataloader.dataset.pipeline = LoadImageFromFile(key='img')

    metric = MagicMock()
    metric.inception_style = 'test_style'
    metric.real_key = 'test_key'
    metric.inception_args = dict(args='test_args')
    cache_tag_1, args_1 = get_inception_feat_cache_name_and_args(
        dataloader, metric, 10, True, True)

    dataloader = MagicMock()
    dataloader.dataset = MagicMock(spec=Dataset)
    dataloader.dataset.data_root = 'test_root'
    dataloader.dataset.data_prefix = 'test_prefix'
    dataloader.dataset.metainfo = dict(meta_info='test_meta')
    dataloader.dataset.pipeline = LoadImageFromFile(key='img')

    metric = MagicMock()
    metric.inception_style = 'test_style'
    metric.real_key = 'test_key'
    metric.inception_args = dict(args='test_args')
    cache_tag_2, args_2 = get_inception_feat_cache_name_and_args(
        dataloader, metric, 10, True, True)
    # check whether cache name are same with the same inputs
    assert cache_tag_1 == cache_tag_2
    assert args_1 == args_2


def test_vgg_feat_cache_name_args():
    dataloader = MagicMock()
    dataloader.dataset = MagicMock(spec=Dataset)
    dataloader.dataset.data_root = 'test_root'
    dataloader.dataset.data_prefix = 'test_prefix'
    dataloader.dataset.metainfo = dict(meta_info='test_meta')
    dataloader.dataset.pipeline = LoadImageFromFile(key='img')

    metric = MagicMock()
    metric.inception_style = 'test_style'
    cache_tag_1, args_1 = get_vgg_feat_cache_name_and_args(dataloader, metric)

    dataloader = MagicMock()
    dataloader.dataset = MagicMock(spec=Dataset)
    dataloader.dataset.data_root = 'test_root'
    dataloader.dataset.data_prefix = 'test_prefix'
    dataloader.dataset.metainfo = dict(meta_info='test_meta')
    dataloader.dataset.pipeline = LoadImageFromFile(key='img')
    cache_tag_2, args_2 = get_vgg_feat_cache_name_and_args(dataloader, metric)

    # check whether cache name are same with the same inputs
    assert cache_tag_1 == cache_tag_2
    assert args_1 == args_2


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
