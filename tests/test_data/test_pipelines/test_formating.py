# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmedit.datasets.pipelines import (Collect, FormatTrimap, GetMaskedImage,
                                       ImageToTensor, ToTensor)
from mmedit.datasets.pipelines.formating import FramesToTensor


def check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


def test_to_tensor():
    to_tensor = ToTensor(['str'])
    with pytest.raises(TypeError):
        results = dict(str='0')
        to_tensor(results)

    target_keys = ['tensor', 'numpy', 'sequence', 'int', 'float']
    to_tensor = ToTensor(target_keys)
    ori_results = dict(
        tensor=torch.randn(2, 3),
        numpy=np.random.randn(2, 3),
        sequence=list(range(10)),
        int=1,
        float=0.1)

    results = to_tensor(ori_results)
    assert check_keys_contain(results.keys(), target_keys)
    for key in target_keys:
        assert isinstance(results[key], torch.Tensor)
        assert torch.equal(results[key].data, ori_results[key])

    # Add an additional key which is not in keys.
    ori_results = dict(
        tensor=torch.randn(2, 3),
        numpy=np.random.randn(2, 3),
        sequence=list(range(10)),
        int=1,
        float=0.1,
        str='test')
    results = to_tensor(ori_results)
    assert check_keys_contain(results.keys(), target_keys)
    for key in target_keys:
        assert isinstance(results[key], torch.Tensor)
        assert torch.equal(results[key].data, ori_results[key])

    assert repr(
        to_tensor) == to_tensor.__class__.__name__ + f'(keys={target_keys})'


def test_image_to_tensor():
    ori_results = dict(img=np.random.randn(256, 256, 3))
    keys = ['img']
    to_float32 = False
    image_to_tensor = ImageToTensor(keys)
    results = image_to_tensor(ori_results)
    assert results['img'].shape == torch.Size([3, 256, 256])
    assert isinstance(results['img'], torch.Tensor)
    assert torch.equal(results['img'].data, ori_results['img'])
    assert results['img'].dtype == torch.float32

    ori_results = dict(img=np.random.randint(256, size=(256, 256)))
    keys = ['img']
    to_float32 = True
    image_to_tensor = ImageToTensor(keys)
    results = image_to_tensor(ori_results)
    assert results['img'].shape == torch.Size([1, 256, 256])
    assert isinstance(results['img'], torch.Tensor)
    assert torch.equal(results['img'].data, ori_results['img'])
    assert results['img'].dtype == torch.float32

    assert repr(image_to_tensor) == (
        image_to_tensor.__class__.__name__ +
        f'(keys={keys}, to_float32={to_float32})')


def test_frames_to_tensor():
    with pytest.raises(TypeError):
        # results[key] should be a list
        ori_results = dict(img=np.random.randn(12, 12, 3))
        FramesToTensor(['img'])(ori_results)

    ori_results = dict(
        img=[np.random.randn(12, 12, 3),
             np.random.randn(12, 12, 3)])
    keys = ['img']
    frames_to_tensor = FramesToTensor(keys, to_float32=False)
    results = frames_to_tensor(ori_results)
    assert results['img'].shape == torch.Size([2, 3, 12, 12])
    assert isinstance(results['img'], torch.Tensor)
    assert torch.equal(results['img'].data[0, ...], ori_results['img'][0])
    assert torch.equal(results['img'].data[1, ...], ori_results['img'][1])
    assert results['img'].dtype == torch.float64

    ori_results = dict(
        img=[np.random.randn(12, 12, 3),
             np.random.randn(12, 12, 3)])
    frames_to_tensor = FramesToTensor(keys, to_float32=True)
    results = frames_to_tensor(ori_results)
    assert results['img'].shape == torch.Size([2, 3, 12, 12])
    assert isinstance(results['img'], torch.Tensor)
    assert torch.equal(results['img'].data[0, ...], ori_results['img'][0])
    assert torch.equal(results['img'].data[1, ...], ori_results['img'][1])
    assert results['img'].dtype == torch.float32

    ori_results = dict(img=[np.random.randn(12, 12), np.random.randn(12, 12)])
    frames_to_tensor = FramesToTensor(keys, to_float32=True)
    results = frames_to_tensor(ori_results)
    assert results['img'].shape == torch.Size([2, 1, 12, 12])
    assert isinstance(results['img'], torch.Tensor)
    assert torch.equal(results['img'].data[0, ...], ori_results['img'][0])
    assert torch.equal(results['img'].data[1, ...], ori_results['img'][1])
    assert results['img'].dtype == torch.float32


def test_masked_img():
    img = np.random.rand(4, 4, 1).astype(np.float32)
    mask = np.zeros((4, 4, 1), dtype=np.float32)
    mask[1, 1] = 1

    results = dict(gt_img=img, mask=mask)
    get_masked_img = GetMaskedImage()
    results = get_masked_img(results)
    masked_img = img * (1. - mask)
    assert np.array_equal(results['masked_img'], masked_img)

    name_ = repr(get_masked_img)
    class_name = get_masked_img.__class__.__name__
    assert name_ == class_name + "(img_name='gt_img', mask_name='mask')"


def test_format_trimap():
    ori_trimap = np.random.randint(3, size=(64, 64))
    ori_trimap[ori_trimap == 1] = 128
    ori_trimap[ori_trimap == 2] = 255

    from mmcv.parallel import DataContainer
    ori_result = dict(
        trimap=torch.from_numpy(ori_trimap.copy()), meta=DataContainer({}))
    format_trimap = FormatTrimap(to_onehot=False)
    results = format_trimap(ori_result)
    result_trimap = results['trimap']
    assert result_trimap.shape == (1, 64, 64)
    assert ((result_trimap.numpy() == 0) == (ori_trimap == 0)).all()
    assert ((result_trimap.numpy() == 1) == (ori_trimap == 128)).all()
    assert ((result_trimap.numpy() == 2) == (ori_trimap == 255)).all()

    ori_result = dict(
        trimap=torch.from_numpy(ori_trimap.copy()), meta=DataContainer({}))
    format_trimap = FormatTrimap(to_onehot=True)
    results = format_trimap(ori_result)
    result_trimap = results['trimap']
    assert result_trimap.shape == (3, 64, 64)
    assert ((result_trimap[0, ...].numpy() == 1) == (ori_trimap == 0)).all()
    assert ((result_trimap[1, ...].numpy() == 1) == (ori_trimap == 128)).all()
    assert ((result_trimap[2, ...].numpy() == 1) == (ori_trimap == 255)).all()

    assert repr(format_trimap) == format_trimap.__class__.__name__ + (
        '(to_onehot=True)')


def test_collect():
    inputs = dict(
        img=np.random.randn(256, 256, 3),
        label=[1],
        img_name='test_image.png',
        ori_shape=(256, 256, 3),
        img_shape=(256, 256, 3),
        pad_shape=(256, 256, 3),
        flip_direction='vertical',
        img_norm_cfg=dict(to_bgr=False))
    keys = ['img', 'label']
    meta_keys = ['img_shape', 'img_name', 'ori_shape']
    collect = Collect(keys, meta_keys=meta_keys)
    results = collect(inputs)
    assert set(list(results.keys())) == set(['img', 'label', 'meta'])
    inputs.pop('img')
    assert set(results['meta'].data.keys()) == set(meta_keys)
    for key in results['meta'].data:
        assert results['meta'].data[key] == inputs[key]

    assert repr(collect) == (
        collect.__class__.__name__ +
        f'(keys={keys}, meta_keys={collect.meta_keys})')
