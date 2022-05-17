# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import mmcv
import numpy as np
import pytest

from mmedit.transforms import LoadImageFromFile


def test_load_image_from_file():

    path_baboon = Path(
        __file__).parent.parent / 'data' / 'image' / 'gt' / 'baboon.png'
    img_baboon = mmcv.imread(str(path_baboon), flag='color')
    h, w, _ = img_baboon.shape

    path_baboon_x4 = Path(
        __file__).parent.parent / 'data' / 'image' / 'lq' / 'baboon_x4.png'
    img_baboon_x4 = mmcv.imread(str(path_baboon_x4), flag='color')

    # read gt image
    # input path is Path object
    results = dict(gt_path=path_baboon)
    config = dict(key='gt')
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results['gt'].shape == (h, w, 3)
    assert results['ori_gt_shape'] == (h, w, 3)
    np.testing.assert_almost_equal(results['gt'], img_baboon)
    assert results['gt_path'] == path_baboon
    # input path is str
    results = dict(gt_path=str(path_baboon))
    results = image_loader(results)
    assert results['gt'].shape == (h, w, 3)
    assert results['ori_gt_shape'] == (h, w, 3)
    np.testing.assert_almost_equal(results['gt'], img_baboon)
    assert results['gt_path'] == str(path_baboon)

    # read input image
    # input path is Path object
    results = dict(img_path=path_baboon_x4)
    config = dict(key='img')
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results['img'].shape == (h // 4, w // 4, 3)
    np.testing.assert_almost_equal(results['img'], img_baboon_x4)
    assert results['img_path'] == path_baboon_x4
    # input path is str
    results = dict(img_path=str(path_baboon_x4))
    results = image_loader(results)
    assert results['img'].shape == (h // 4, w // 4, 3)
    np.testing.assert_almost_equal(results['img'], img_baboon_x4)
    assert results['img_path'] == str(path_baboon_x4)
    assert repr(image_loader) == (
        image_loader.__class__.__name__ +
        ('(key=img, color_type=color, channel_order=bgr, '
         'imdecode_backend=cv2, use_cache=False, to_float32=False, '
         'to_y_channel=False, save_original_img=False, '
         "file_client_args={'backend': 'disk'})"))

    # test save_original_img
    results = dict(img_path=path_baboon)
    config = dict(key='img', color_type='grayscale', save_original_img=True)
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results['img'].shape == (h, w)
    assert results['ori_img_shape'] == (h, w)
    np.testing.assert_almost_equal(results['ori_img'], results['img'])
    assert id(results['ori_img']) != id(results['img'])

    # test: use_cache
    results = dict(gt_path=path_baboon)
    config = dict(key='gt', use_cache=True)
    image_loader = LoadImageFromFile(**config)
    assert image_loader.cache == dict()
    assert repr(image_loader) == (
        image_loader.__class__.__name__ +
        ('(key=gt, color_type=color, channel_order=bgr, '
         'imdecode_backend=cv2, use_cache=True, to_float32=False, '
         'to_y_channel=False, save_original_img=False, '
         "file_client_args={'backend': 'disk'})"))
    results = image_loader(results)
    assert image_loader.cache is not None
    assert str(path_baboon) in image_loader.cache
    assert results['gt'].shape == (h, w, 3)
    assert results['gt_path'] == path_baboon
    np.testing.assert_almost_equal(results['gt'], img_baboon)

    # convert to y-channel (bgr2y)
    results = dict(gt_path=path_baboon)
    config = dict(key='gt', to_y_channel=True, to_float32=True)
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results['gt'].shape == (h, w, 1)
    img_baboon_y = mmcv.bgr2ycbcr(img_baboon, y_only=True)
    img_baboon_y = np.expand_dims(img_baboon_y, axis=2)
    np.testing.assert_almost_equal(results['gt'], img_baboon_y)
    assert results['gt_path'] == path_baboon

    # convert to y-channel (rgb2y)
    results = dict(gt_path=path_baboon)
    config = dict(
        key='gt', channel_order='rgb', to_y_channel=True, to_float32=True)
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results['gt'].shape == (h, w, 1)
    img_baboon_y = mmcv.bgr2ycbcr(img_baboon, y_only=True)
    img_baboon_y = np.expand_dims(img_baboon_y, axis=2)
    np.testing.assert_almost_equal(results['gt'], img_baboon_y)

    # test frames
    # input path is Path object
    results = dict(gt_path=[path_baboon])
    config = dict(key='gt', save_original_img=True)
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results['gt'][0].shape == (h, w, 3)
    assert results['ori_gt_shape'] == [(h, w, 3)]
    np.testing.assert_almost_equal(results['gt'][0], img_baboon)
    assert results['gt_path'] == [path_baboon]
    # input path is str
    results = dict(gt_path=[str(path_baboon)])
    results = image_loader(results)
    assert results['gt'][0].shape == (h, w, 3)
    assert results['ori_gt_shape'] == [(h, w, 3)]
    np.testing.assert_almost_equal(results['gt'][0], img_baboon)
    assert results['gt_path'] == [str(path_baboon)]

    # test lmdb
    results = dict(img_path=path_baboon)
    config = dict(
        key='img',
        file_client_args=dict(
            backend='lmdb',
            db_path=Path(__file__).parent.parent / 'data' / 'lq.lmdb'))
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results['img'].shape == (h // 4, w // 4, 3)
    assert results['ori_img_shape'] == (h // 4, w // 4, 3)
    assert results['img_path'] == path_baboon

    # convert to y-channel (ValueError)
    results = dict(gt_path=path_baboon)
    config = dict(key='gt', to_y_channel=True, channel_order='gg')
    image_loader = LoadImageFromFile(**config)
    with pytest.raises(ValueError):
        results = image_loader(results)
