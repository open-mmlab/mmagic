# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import mmcv
import numpy as np
import pytest
from mmengine.fileio.backends import LocalBackend

from mmagic.datasets.transforms import (GetSpatialDiscountMask,
                                        LoadImageFromFile, LoadMask)


def test_load_image_from_file():

    path_baboon = Path(
        __file__).parent.parent.parent / 'data' / 'image' / 'gt' / 'baboon.png'
    img_baboon = mmcv.imread(str(path_baboon), flag='color')
    h, w, _ = img_baboon.shape

    path_baboon_x4 = Path(
        __file__
    ).parent.parent.parent / 'data' / 'image' / 'lq' / 'baboon_x4.png'
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
    assert results['gt_channel_order'] == 'bgr'
    assert results['gt_color_type'] == 'color'

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
         'imdecode_backend=None, use_cache=False, to_float32=False, '
         'to_y_channel=False, save_original_img=False, '
         'backend_args=None)'))
    assert isinstance(image_loader.file_backend, LocalBackend)

    # test save_original_img
    results = dict(img_path=path_baboon)
    config = dict(key='img', color_type='grayscale', save_original_img=True)
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results['img'].shape == (h, w, 1)
    assert results['ori_img_shape'] == (h, w, 1)
    np.testing.assert_almost_equal(results['ori_img'], results['img'])
    assert id(results['ori_img']) != id(results['img'])
    assert results['img_channel_order'] == 'bgr'
    assert results['img_color_type'] == 'grayscale'

    # test: use_cache
    results = dict(gt_path=path_baboon)
    config = dict(key='gt', use_cache=True)
    image_loader = LoadImageFromFile(**config)
    assert image_loader.cache == dict()
    assert repr(image_loader) == (
        image_loader.__class__.__name__ +
        ('(key=gt, color_type=color, channel_order=bgr, '
         'imdecode_backend=None, use_cache=True, to_float32=False, '
         'to_y_channel=False, save_original_img=False, '
         'backend_args=None)'))
    results = image_loader(results)
    assert image_loader.cache is not None
    assert str(path_baboon) in image_loader.cache
    assert results['gt'].shape == (h, w, 3)
    assert results['gt_path'] == path_baboon
    np.testing.assert_almost_equal(results['gt'], img_baboon)
    assert isinstance(image_loader.file_backend, LocalBackend)
    assert results['gt_channel_order'] == 'bgr'
    assert results['gt_color_type'] == 'color'

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
    assert results['gt_channel_order'] == 'bgr'
    assert results['gt_color_type'] == 'color'

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
        backend_args=dict(
            backend='lmdb',
            db_path=Path(__file__).parent.parent.parent / 'data' / 'lq.lmdb'))
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

    # test infer from s3
    from mmengine.fileio import PetrelBackend
    PetrelBackend.get = lambda self, filepath: filepath
    results = dict(img_path='openmmlab:s3://abcd/efg/')
    config = dict(key='img')
    image_loader = LoadImageFromFile(**config)

    try:
        import petrel_client  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError) as excinfo:
            results = image_loader(results)
        assert 'Please install petrel_client to enable PetrelBackend.' in str(
            excinfo.value)
    else:
        results = image_loader(results)
        assert results['img'] == 'openmmlab:s3://abcd/efg/'


def test_dct_mask():
    mask = np.zeros((64, 64, 1))
    mask[20:40, 20:40] = 1.
    mask_bbox = [20, 20, 20, 20]
    results = dict(mask=mask, mask_bbox=mask_bbox)

    dct_mask = GetSpatialDiscountMask()
    results = dct_mask(results)
    assert 'discount_mask' in results
    assert results['discount_mask'].shape == (64, 64, 1)

    mask_height = mask_width = 20
    gamma = 0.99
    beta = 1.5
    mask_values = np.ones((mask_width, mask_height, 1))
    for i in range(mask_width):
        for j in range(mask_height):
            mask_values[i,
                        j] = max(gamma**(min(i, mask_width - i - 1) * beta),
                                 gamma**(min(j, mask_height - j - 1) * beta))
    dct_mask_test = np.zeros_like(mask)
    dct_mask_test[20:40, 20:40, ...] = mask_values

    np.testing.assert_almost_equal(dct_mask_test, results['discount_mask'])
    repr_str = dct_mask.__class__.__name__ + (f'(gamma={dct_mask.gamma}, '
                                              f'beta={dct_mask.beta})')
    assert repr_str == repr(dct_mask)


class TestInpaintLoading:

    @classmethod
    def setup_class(cls):
        cls.img_path = Path(__file__).parent.parent.parent.joinpath(
            'data/image/test.png')
        cls.results = dict(img_info=dict(filename=cls.img_path))

    def test_load_mask(self):

        # test mask mode: set
        mask_config = dict(
            mask_list_file='tests/data/inpainting/mask_list.txt',
            prefix='tests/data/inpainting/',
            io_backend='local',
            color_type='unchanged',
            file_client_kwargs=dict())

        set_loader = LoadMask('set', mask_config)
        class_name = set_loader.__class__.__name__
        assert repr(set_loader) == class_name + "(mask_mode='set')"
        for _ in range(2):
            results = dict()
            results = set_loader(results)
            gt_mask = mmcv.imread(
                'tests/data/inpainting/mask/test.png', flag='unchanged')
            assert np.array_equal(results['mask'], gt_mask[..., 0:1] / 255.)

        mask_config = dict(
            mask_list_file='tests/data/inpainting/mask_list_single_ch.txt',
            prefix='tests/data/inpainting/',
            io_backend='local',
            color_type='unchanged',
            file_client_kwargs=dict())

        # test mask mode: set with input as single channel image
        set_loader = LoadMask('set', mask_config)
        results = dict()
        results = set_loader(results)
        gt_mask = mmcv.imread(
            'tests/data/inpainting/mask/test_single_ch.png', flag='unchanged')
        gt_mask = np.expand_dims(gt_mask, axis=2)
        assert np.array_equal(results['mask'], gt_mask[..., 0:1] / 255.)

        # test mask mode: ff
        mask_config = dict(
            img_shape=(256, 256),
            num_vertices=(4, 12),
            mean_angle=1.2,
            angle_range=0.4,
            brush_width=(12, 40))

        ff_loader = LoadMask('ff', mask_config)
        results = dict()
        results = ff_loader(results)
        assert results['mask'].shape == (256, 256, 1)

        # test mask mode: irregular holes
        mask_config = dict(
            img_shape=(256, 256),
            num_vertices=(4, 12),
            max_angle=4.,
            length_range=(10, 100),
            brush_width=(10, 40),
            area_ratio_range=(0.15, 0.5))

        irregular_loader = LoadMask('irregular', mask_config)
        results = dict()
        results = irregular_loader(results)
        assert results['mask'].shape == (256, 256, 1)

        # test mask mode: bbox
        mask_config = dict(img_shape=(256, 256), max_bbox_shape=128)

        bbox_loader = LoadMask('bbox', mask_config)
        results = dict()
        results = bbox_loader(results)
        assert results['mask'].shape == (256, 256, 1)

        # test mask mode: file
        mask_loader = LoadMask('file')
        mask = mask_loader(
            dict(mask_path='tests/data/inpainting/mask/test_single_ch.png'))
        assert mask['mask'].shape == (256, 256, 1)

        with pytest.raises(NotImplementedError):
            loader = LoadMask('xxxx', mask_config)
            results = loader(results)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
