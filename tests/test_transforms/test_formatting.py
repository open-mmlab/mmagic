# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import to_tensor

from mmedit.data_element.edit_data_sample import EditDataSample
from mmedit.transforms import PackEditInputs, ToTensor
from mmedit.transforms.formatting import images_to_tensor


def test_images_to_tensor():

    data = [np.random.rand(64, 64, 3), np.random.rand(64, 64, 3)]
    tensor = images_to_tensor(data)
    assert tensor.shape == torch.Size([2, 3, 64, 64])

    data = np.random.rand(64, 64, 3)
    tensor = images_to_tensor(data)
    assert tensor.shape == torch.Size([3, 64, 64])

    data = 1
    tensor = images_to_tensor(data)
    assert tensor == torch.tensor(1)


def assert_tensor_equal(img, ref_img, ratio_thr=0.999):
    """Check if img and ref_img are matched approximately."""
    assert img.shape == ref_img.shape
    assert img.dtype == ref_img.dtype
    area = ref_img.shape[-1] * ref_img.shape[-2]
    diff = torch.abs(img - ref_img)
    assert torch.sum(diff <= 1) / float(area) > ratio_thr


def test_pack_edit_inputs():

    pack_edit_inputs = PackEditInputs()
    assert repr(pack_edit_inputs) == 'PackEditInputs'

    ori_results = dict(
        img=np.random.rand(64, 64, 3),
        gt=[np.random.rand(64, 61, 3),
            np.random.rand(64, 61, 3)],
        ref=np.random.rand(64, 62, 3),
        mask=np.random.rand(64, 63, 3),
        gt_heatmap=np.random.rand(64, 65, 3),
        trimap=np.random.rand(64, 66, 3),
        gt_alpha=np.random.rand(64, 67, 3),
        gt_fg=np.random.rand(64, 68, 3),
        gt_bg=np.random.rand(64, 69, 3),
        img_shape=(64, 64),
        a='b')

    results = ori_results.copy()

    packed_results = pack_edit_inputs(results)

    target_keys = ['inputs', 'data_sample']
    assert set(target_keys).issubset(set(packed_results.keys()))

    data_sample = packed_results['data_sample']
    assert isinstance(data_sample, EditDataSample)

    gt_tensors = [to_tensor(v) for v in ori_results['gt']]
    gt_tensors = [v.permute(2, 0, 1) for v in gt_tensors]
    gt_tensor = torch.stack(gt_tensors, dim=0)
    assert_tensor_equal(data_sample.gt_img.data, gt_tensor)

    ref_tensor = to_tensor(ori_results['ref'])
    ref_tensor = ref_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.ref_img.data, ref_tensor)

    mask_tensor = to_tensor(ori_results['mask'])
    mask_tensor = mask_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.mask.data, mask_tensor)

    gt_heatmap_tensor = to_tensor(ori_results['gt_heatmap'])
    gt_heatmap_tensor = gt_heatmap_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.gt_heatmap.data, gt_heatmap_tensor)

    trimap_tensor = to_tensor(ori_results['trimap'])
    trimap_tensor = trimap_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.trimap.data, trimap_tensor)

    gt_alpha_tensor = to_tensor(ori_results['gt_alpha'])
    gt_alpha_tensor = gt_alpha_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.gt_alpha.data, gt_alpha_tensor)

    gt_fg_tensor = to_tensor(ori_results['gt_fg'])
    gt_fg_tensor = gt_fg_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.gt_fg.data, gt_fg_tensor)

    gt_bg_tensor = to_tensor(ori_results['gt_bg'])
    gt_bg_tensor = gt_bg_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.gt_bg.data, gt_bg_tensor)

    assert data_sample.metainfo['img_shape'] == (64, 64)
    assert data_sample.metainfo['a'] == 'b'


def test_to_tensor():

    ori_results = dict(
        img=np.random.rand(64, 64, 3),
        gt=[np.random.rand(64, 64, 3),
            np.random.rand(64, 64, 3)],
        a=1)

    keys = ['img', 'gt', 'a']
    to_tensor = ToTensor(keys=keys, to_float32=True)
    assert repr(to_tensor) == f'ToTensor(keys={keys}, to_float32=True)'

    results = to_tensor(ori_results)
    assert set(keys).issubset(results.keys())
    for _, v in results.items():
        assert isinstance(v, torch.Tensor)
