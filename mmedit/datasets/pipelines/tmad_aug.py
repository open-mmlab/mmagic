import logging

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.utils import print_log

from ..registry import PIPELINES


@PIPELINES.register_module()
class GetPatchPool(object):
    """Get patch pool of valid region and mask regions.

    In this augmentation, the input images will be unfolded to multiple patches
    and re-organized as a patch pool.

    Different from other common used pipelines, the input here should be
    an instance of torch.Tensor. As for the valid region (unmasked region), the
    patch pool will be returned directly while an index indicator will be
    returned for hole region (masked region) to extract masked patches in
    inpainting models.

    Args:
        gt_img_key (str): The ground-truth or masked images to be unfolded. In
            training, we use ground-truth image as default. But in testing, we
            can only obtain the masked image.
        mask_key (str): The mask image to be unfolded.
        patch_size (int | tuple[int]): Same as `torch.nn.Unfold`. The size of
            patches in the pool.
        stride_valid (int | tuple[int]): Same as `torch.nn.Unfold`. The stride
            will be used in unfolding valid (unmasked) regions.
        stride_hole (int | tuple[int]): Same as `torch.nn.Unfold`. The stride
            will be used in unfolding hole (masked) regions.
        valid_mask_thr (float): The threshold for the masked region ratio in
            valid (unmasked) region.
        hole_mask_thr (float): The threshold for the masked region ratio in
            hole (masked) region.
        num_pool_hole (int): The number of the patches in pools of hole region.
        num_pool_valid (int): The number of the patches in pools of valid
            region.
        mode (str): The mode should be one of ['train', 'test'].
            Default: 'train'.
    """

    def __init__(self,
                 gt_img_key,
                 mask_key,
                 patch_size,
                 stride_valid,
                 stride_hole,
                 valid_mask_thr,
                 hole_mask_thr,
                 num_pool_hole,
                 num_pool_valid,
                 mode='train'):
        self.gt_img_key = gt_img_key
        self.mask_key = mask_key
        self.patch_size = patch_size
        self.stride_valid = stride_valid
        self.stride_hole = stride_hole
        self.valid_mask_thr = valid_mask_thr
        self.hole_mask_thr = hole_mask_thr
        self.num_pool_hole = num_pool_hole
        self.num_pool_valid = num_pool_valid
        self.mode = mode

    def __call__(self, results):
        gt_img = results[self.gt_img_key]
        mask = results[self.mask_key]

        # get patch pool for valid (unmasked) regions
        gt_img_pool = self.im2col(gt_img, self.patch_size,
                                  self.stride_valid)[0]
        mask_pool = self.im2col(mask, self.patch_size, self.stride_valid)[0]
        score_pool = torch.mean(mask_pool, dim=(1, 2, 3))
        # index indicator for which patches to be chosed
        valid_index = torch.le(score_pool, self.valid_mask_thr)

        # select patches as valid patch
        valid_patch_pool = gt_img_pool[valid_index, ...]
        valid_mask_patch_pool = mask_pool[valid_index, ...]

        num_valid_patches = valid_patch_pool.size(0)

        # append existing patches if the patches is not enough to build a pool
        if num_valid_patches < self.num_pool_valid:
            diff_ = self.num_pool_valid - num_valid_patches
            repeat_times = diff_ // num_valid_patches + 2
            valid_patch_pool = valid_patch_pool.repeat(repeat_times, 1, 1, 1)
            valid_mask_patch_pool = valid_mask_patch_pool.repeat(
                repeat_times, 1, 1, 1)

        # sample patches from current (enlarged) pool
        sampled_valid_pool, sample_list_valid = self.sample_patches(
            valid_patch_pool, self.num_pool_valid)
        sampled_mask_valid_pool, _ = self.sample_patches(
            valid_mask_patch_pool, self.num_pool_valid, sample_list_valid)

        # get the index indicator for masked regions
        hole_patch_index = self._get_hole_patch_index(mask, self.hole_mask_thr)

        results['valid_patch_pool'] = sampled_valid_pool
        results['valid_mask_patch_pool'] = sampled_mask_valid_pool
        results['valid_score_pool'] = score_pool
        results['hole_patch_index'] = hole_patch_index

        return results

    def _get_hole_patch_index(self, mask, mask_thr=0.):
        """Get index indicator for patches in the hole.

        In this function, we will get the patch index for the hole region that
        we will use to construct patch pool in the model.

        Args:
            mask (torch.Tensor): Mask tensor with shape of (n, c, h, w) or
                (c, h, w)
            mask_thr (float, optional): Threshold for getting valid hole
                patches. Defaults to 0..

        Returns:
            torch.Tensor: Tensor with shape of (l, ), where l is the size of
                the pool.
        """

        # construct mask pool for hole regions
        mask_pool = self.im2col(mask, self.patch_size, self.stride_hole)[0]
        mask_mean = torch.mean(mask_pool, dim=(1, 2, 3))
        # use the mask_thr to get the index indicator
        hole_patch_index = torch.gt(mask_mean, mask_thr)
        num_hole_patches = torch.sum(hole_patch_index.int()).item()

        # appending patches from valid regions
        if num_hole_patches < self.num_pool_hole:
            index_list = np.arange(mask_pool.size(0), dtype=np.int32)
            unchosed_index = ~hole_patch_index.numpy().astype(np.bool)
            unchosed_list = index_list[unchosed_index]
            extra_index_list = np.random.choice(
                unchosed_list,
                self.num_pool_hole - num_hole_patches,
                replace=False)
            hole_patch_index_final = hole_patch_index.clone()
            hole_patch_index_final[extra_index_list] = True
        # just ignore some patches if the pool is full and in this version, we
        # simply ignore the last several patches
        elif num_hole_patches > self.num_pool_hole:
            if self.mode == 'train':
                print_log(
                    f'The number of hole patches {num_hole_patches} > '
                    f'{self.num_pool_hole}. If in training mode, you can'
                    'just ignore this warning. However, if in testing mode, '
                    'you must set [num_pool_hole] with bigger number',
                    logger='root',
                    level=logging.WARNING)
            else:
                raise ValueError(
                    f'The number of hole patches {num_hole_patches} > '
                    f'{self.num_pool_hole}. In testing mode, '
                    'you must set [num_pool_hole] with bigger number')

            hole_counter_ = 0

            for index_num in range(mask_pool.size(0)):
                if hole_patch_index[index_num]:
                    hole_counter_ += 1
                if hole_counter_ == self.num_pool_hole:
                    break
            hole_patch_index_final = hole_patch_index.clone()
            hole_patch_index_final[index_num + 1:] = False
        else:
            hole_patch_index_final = hole_patch_index.clone()

        return hole_patch_index_final

    def sample_patches(self, pool, num_samples, sample_list=None):
        """Sample patches from the enlarged pool.

        In this function, a sampled pool will be returned according to
        `num_smaples` or `sample_list`.

        Args:
            pool (torch.Tensor): Tensor with shape of (l', c, h, w).
            num_samples (int): The number of smaples that we need.
            sample_list (list[bool], optional): The indicator list for chosed
                samples. Once `sample_list` is given, `num_samples` will be
                ignored. The shape of `sample_list` should be (l', ).
                Defaults to None.

        Returns:
            tuple(torch.Tensor): Sampled pool as tensor with shape of
                (l, c, h, w) and sample list as tensor with shape of (l', ).
        """
        num_pool = pool.size(0)
        if sample_list is None:
            if num_samples > num_pool:
                raise ValueError(f'num_samples ({num_samples}) > num_pool'
                                 f' ({num_pool}).')
            sample_list = sorted(
                np.random.choice(num_pool, num_samples, replace=False))

        sampled_pool = pool[sample_list, ...]

        return sampled_pool, sample_list

    def im2col(self, img, kernel_size, stride):
        """Image to columns.

        Different from `torch.nn.Unfold`, the returned tensor will be reshaped
        to (n, p, c, h, w).

        Args:
            img (torch.Tensor): Image tensor with shape of (n, c, h, w) or
                (c, h, w).
            kernel_size (int | tuple[int]): Same as `torch.nn.Unfold`.
            stride (int | tuple[int]): Same as `torch.nn.Unfold`.

        Returns:
            torch.Tensor: Image columns (patches) with shape of (n, p, c, h, w)
                where 'p' is the number of columns or patches.
        """
        assert img.dim() == 3 or img.dim() == 4, (
            'The shape of image tensor should be (n, c, h, w) or (c, h, w)')
        if img.dim() == 3:
            img = img.view(1, *img.size())
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        img_unfold = F.unfold(img, kernel_size, stride=stride)
        img_unfold = img_unfold.permute(0, 2, 1)
        num_batch, num_patch = img_unfold.size()[:2]
        img_reshaped = img_unfold.view(num_batch, num_patch, img.size(1),
                                       kernel_size[0], kernel_size[1])

        return img_reshaped

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f' (gt_img_key={self.gt_img_key}, mask_key={self.mask_key}, '
            f'patch_size={self.patch_size}, stride_valid={self.stride_valid},'
            f' stride_hole={self.stride_hole}, '
            f'valid_mask_thr={self.valid_mask_thr}, '
            f'hole_mask_thr={self.hole_mask_thr}, num_pool_hole='
            f'{self.num_pool_hole}, num_pool_valid={self.num_pool_valid})')

        return repr_str
