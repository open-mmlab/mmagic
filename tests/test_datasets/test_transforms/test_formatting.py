# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import to_tensor

from mmagic.datasets.transforms import PackInputs
from mmagic.structures.data_sample import DataSample


def assert_tensor_equal(img, ref_img, ratio_thr=0.999):
    """Check if img and ref_img are matched approximately."""
    assert img.shape == ref_img.shape
    assert img.dtype == ref_img.dtype
    area = ref_img.shape[-1] * ref_img.shape[-2]
    diff = torch.abs(img - ref_img)
    assert torch.sum(diff <= 1) / float(area) > ratio_thr


def test_pack_inputs():

    pack_inputs = PackInputs(meta_keys='a', data_keys='numpy')
    assert repr(pack_inputs) == 'PackInputs'

    ori_results = dict(
        img=np.random.rand(64, 64, 3),
        gt=[np.random.rand(64, 61, 3),
            np.random.rand(64, 61, 3)],
        img_lq=np.random.rand(64, 64, 3),
        ref=np.random.rand(64, 62, 3),
        ref_lq=np.random.rand(64, 62, 3),
        mask=np.random.rand(64, 63, 3),
        gt_heatmap=np.random.rand(64, 65, 3),
        gt_unsharp=np.random.rand(64, 65, 3),
        merged=np.random.rand(64, 64, 3),
        trimap=np.random.rand(64, 66, 3),
        alpha=np.random.rand(64, 67, 3),
        fg=np.random.rand(64, 68, 3),
        bg=np.random.rand(64, 69, 3),
        img_shape=(64, 64),
        a='b',
        numpy=np.random.rand(48, 48, 3))

    results = ori_results.copy()

    packed_results = pack_inputs(results)

    target_keys = ['inputs', 'data_samples']
    assert set(target_keys).issubset(set(packed_results.keys()))

    data_sample = packed_results['data_samples']
    assert isinstance(data_sample, DataSample)

    assert data_sample.img_shape == (64, 64)
    assert data_sample.a == 'b'

    numpy_tensor = to_tensor(ori_results['numpy'])
    numpy_tensor = numpy_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.numpy, numpy_tensor)

    gt_tensors = [to_tensor(v) for v in ori_results['gt']]
    gt_tensors = [v.permute(2, 0, 1) for v in gt_tensors]
    gt_tensor = torch.stack(gt_tensors, dim=0)
    assert_tensor_equal(data_sample.gt_img, gt_tensor)

    img_lq_tensor = to_tensor(ori_results['ref'])
    img_lq_tensor = img_lq_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.ref_img, img_lq_tensor)

    ref_lq_tensor = to_tensor(ori_results['ref'])
    ref_lq_tensor = ref_lq_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.ref_img, ref_lq_tensor)

    ref_tensor = to_tensor(ori_results['ref'])
    ref_tensor = ref_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.ref_img, ref_tensor)

    mask_tensor = to_tensor(ori_results['mask'])
    mask_tensor = mask_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.mask, mask_tensor)

    gt_heatmap_tensor = to_tensor(ori_results['gt_heatmap'])
    gt_heatmap_tensor = gt_heatmap_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.gt_heatmap, gt_heatmap_tensor)

    gt_unsharp_tensor = to_tensor(ori_results['gt_heatmap'])
    gt_unsharp_tensor = gt_unsharp_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.gt_heatmap, gt_unsharp_tensor)

    gt_merged_tensor = to_tensor(ori_results['merged'])
    gt_merged_tensor = gt_merged_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.gt_merged, gt_merged_tensor)

    trimap_tensor = to_tensor(ori_results['trimap'])
    trimap_tensor = trimap_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.trimap, trimap_tensor)

    gt_alpha_tensor = to_tensor(ori_results['alpha'])
    gt_alpha_tensor = gt_alpha_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.gt_alpha, gt_alpha_tensor)

    gt_fg_tensor = to_tensor(ori_results['fg'])
    gt_fg_tensor = gt_fg_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.gt_fg, gt_fg_tensor)

    gt_bg_tensor = to_tensor(ori_results['bg'])
    gt_bg_tensor = gt_bg_tensor.permute(2, 0, 1)
    assert_tensor_equal(data_sample.gt_bg, gt_bg_tensor)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
