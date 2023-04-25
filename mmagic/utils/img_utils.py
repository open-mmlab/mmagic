# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Tuple

import numpy as np
import torch
from mmcv.transforms import to_tensor
from torchvision.utils import make_grid


def can_convert_to_image(value):
    """Judge whether the input value can be converted to image tensor via
    :func:`images_to_tensor` function.

    Args:
        value (any): The input value.

    Returns:
        bool: If true, the input value can convert to image with
            :func:`images_to_tensor`, and vice versa.
    """
    if isinstance(value, (List, Tuple)):
        return all([can_convert_to_image(v) for v in value])
    elif isinstance(value, np.ndarray) and len(value.shape) > 1:
        return True
    elif isinstance(value, torch.Tensor):
        return True
    else:
        return False


def image_to_tensor(img):
    """Trans image to tensor.

    Args:
        img (np.ndarray): The original image.

    Returns:
        Tensor: The output tensor.
    """

    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = np.ascontiguousarray(img)
    tensor = to_tensor(img).permute(2, 0, 1).contiguous()

    return tensor


def all_to_tensor(value):
    """Trans image and sequence of frames to tensor.

    Args:
        value (np.ndarray | list[np.ndarray] | Tuple[np.ndarray]):
            The original image or list of frames.

    Returns:
        Tensor: The output tensor.
    """

    if not can_convert_to_image(value):
        return value

    if isinstance(value, (List, Tuple)):
        # sequence of frames
        if len(value) == 1:
            tensor = image_to_tensor(value[0])
        else:
            frames = [image_to_tensor(v) for v in value]
            tensor = torch.stack(frames, dim=0)
    elif isinstance(value, np.ndarray):
        tensor = image_to_tensor(value)
    else:
        # Maybe the data has been converted to Tensor.
        tensor = to_tensor(value)

    return tensor


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to (min, max), image values will be normalized to [0, 1].

    For different tensor shapes, this function will have different behaviors:

        1. 4D mini-batch Tensor of shape (N x 3/1 x H x W):
            Use `make_grid` to stitch images in the batch dimension, and then
            convert it to numpy array.
        2. 3D Tensor of shape (3/1 x H x W) and 2D Tensor of shape (H x W):
            Directly change to numpy array.

    Note that the image channel in input tensors should be RGB order. This
    function will convert it to cv2 convention, i.e., (H x W x C) with BGR
    order.

    Args:
        tensor (Tensor | list[Tensor]): Input tensors.
        out_type (numpy type): Output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple): min and max values for clamp.

    Returns:
        (Tensor | list[Tensor]): 3D ndarray of shape (H x W x C) or 2D ndarray
        of shape (H x W).
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        # Squeeze two times so that:
        # 1. (1, 1, h, w) -> (h, w) or
        # 3. (1, 3, h, w) -> (3, h, w) or
        # 2. (n>1, 3/1, h, w) -> (n>1, 3/1, h, w)
        _tensor = _tensor.squeeze(0).squeeze(0)
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise ValueError('Only support 4D, 3D or 2D tensor. '
                             f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (np.ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        np.ndarray: Reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        if isinstance(img, np.ndarray):
            img = img.transpose(1, 2, 0)
        elif isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)
    return img


def to_numpy(img, dtype=np.float64):
    """Convert data into numpy arrays of dtype.

    Args:
        img (Tensor | np.ndarray): Input data.
        dtype (np.dtype): Set the data type of the output. Default: np.float64

    Returns:
        img (np.ndarray): Converted numpy arrays data.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    elif not isinstance(img, np.ndarray):
        raise TypeError('Only support torch.tensor and np.ndarray, '
                        f'but got type {type(img)}')

    img = img.astype(dtype)

    return img


def get_box_info(pred_bbox, original_shape, final_size):
    """

    Args:
        pred_bbox: The bounding box for the instance
        original_shape: Original image shape
        final_size: Size of the final output

    Returns:
        List: [L_pad, R_pad, T_pad, B_pad, rh, rw]
    """
    assert len(pred_bbox) == 4
    resize_startx = int(pred_bbox[0] / original_shape[0] * final_size)
    resize_starty = int(pred_bbox[1] / original_shape[1] * final_size)
    resize_endx = int(pred_bbox[2] / original_shape[0] * final_size)
    resize_endy = int(pred_bbox[3] / original_shape[1] * final_size)
    rh = resize_endx - resize_startx
    rw = resize_endy - resize_starty
    if rh < 1:
        if final_size - resize_endx > 1:
            resize_endx += 1
        else:
            resize_startx -= 1
        rh = 1
    if rw < 1:
        if final_size - resize_endy > 1:
            resize_endy += 1
        else:
            resize_starty -= 1
        rw = 1
    L_pad = resize_startx
    R_pad = final_size - resize_endx
    T_pad = resize_starty
    B_pad = final_size - resize_endy
    return [L_pad, R_pad, T_pad, B_pad, rh, rw]
