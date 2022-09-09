# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def extract_bbox_patch(bbox, img, channel_first=True):
    """Extract patch from a given bbox.

    Args:
        bbox (torch.Tensor | numpy.array): Bbox with (top, left, h, w). If
            `img` has batch dimension, the `bbox` must be stacked at first
            dimension. The shape should be (4,) or (n, 4).
        img (torch.Tensor | numpy.array): Image data to be extracted. If
            organized in batch dimension, the batch dimension must be the first
            order like (n, h, w, c) or (n, c, h, w).
        channel_first (bool): If True, the channel dimension of img is before
            height and width, e.g. (c, h, w). Otherwise, the img shape (samples
            in the batch) is like (h, w, c).

    Returns:
        (torch.Tensor | numpy.array): Extracted patches. The dimension of the \
            output should be the same as `img`.
    """

    def _extract(bbox, img):
        assert len(bbox) == 4
        t, l, h, w = bbox
        if channel_first:
            img_patch = img[..., t:t + h, l:l + w]
        else:
            img_patch = img[t:t + h, l:l + w, ...]

        return img_patch

    input_size = img.shape
    assert len(input_size) == 3 or len(input_size) == 4
    bbox_size = bbox.shape
    assert bbox_size == (4, ) or (len(bbox_size) == 2
                                  and bbox_size[0] == input_size[0])

    # images with batch dimension
    if len(input_size) == 4:
        output_list = []
        for i in range(input_size[0]):
            img_patch_ = _extract(bbox[i], img[i:i + 1, ...])
            output_list.append(img_patch_)
        if isinstance(img, torch.Tensor):
            img_patch = torch.cat(output_list, dim=0)
        else:
            img_patch = np.concatenate(output_list, axis=0)
    # standardize image
    else:
        img_patch = _extract(bbox, img)

    return img_patch


def scale_bbox(bbox, target_size):
    """Modify bbox to target size.

    The original bbox will be enlarged to the target size with the original
    bbox in the center of the new bbox.

    Args:
        bbox (np.ndarray | torch.Tensor): Bboxes to be modified. Bbox can
            be in batch or not. The shape should be (4,) or (n, 4).
        target_size (tuple[int]): Target size of final bbox.

    Returns:
        (np.ndarray | torch.Tensor): Modified bboxes.
    """

    def _mod(bbox, target_size):
        top_ori, left_ori, h_ori, w_ori = bbox
        h, w = target_size
        assert h >= h_ori and w >= w_ori
        top = int(max(0, top_ori - (h - h_ori) // 2))
        left = int(max(0, left_ori - (w - w_ori) // 2))

        if isinstance(bbox, torch.Tensor):
            bbox_new = torch.Tensor([top, left, h, w]).type_as(bbox)
        else:
            bbox_new = np.asarray([top, left, h, w])

        return bbox_new

    if isinstance(bbox, torch.Tensor):
        bbox_new = torch.zeros_like(bbox)
    elif isinstance(bbox, np.ndarray):
        bbox_new = np.zeros_like(bbox)
    else:
        raise TypeError('bbox mush be torch.Tensor or numpy.ndarray'
                        f'but got type {type(bbox)}')
    bbox_shape = list(bbox.shape)

    if len(bbox_shape) == 2:
        for i in range(bbox_shape[0]):
            bbox_new[i, :] = _mod(bbox[i], target_size)
    else:
        bbox_new = _mod(bbox, target_size)

    return bbox_new


def extract_around_bbox(img, bbox, target_size, channel_first=True):
    """Extract patches around the given bbox.

    Args:
        bbox (np.ndarray | torch.Tensor): Bboxes to be modified. Bbox can
            be in batch or not.
        target_size (List(int)): Target size of final bbox.

    Returns:
        (torch.Tensor | numpy.array): Extracted patches. The dimension of the \
            output should be the same as `img`.
    """
    bbox_new = scale_bbox(bbox, target_size)
    img_patch = extract_bbox_patch(bbox_new, img, channel_first=channel_first)

    return img_patch, bbox_new
