# This code is referenced from matlab_imresize with modifications
# Reference: https://github.com/fatheral/matlab_imresize/blob/master/imresize.py  # noqa
# Original licence: Copyright (c) 2020 fatheral, under the MIT License.
import numpy as np

from ..registry import PIPELINES


def get_size_from_scale(input_size, scale_factor):
    """Get the output size given input size and scale factor.

    Args:
        input_size (tuple): The size of the input image.
        scale_factor (float): The resize factor.

    Returns:
        list[int]: The size of the output image.
    """

    output_shape = [
        int(np.ceil(scale * shape))
        for (scale, shape) in zip(scale_factor, input_size)
    ]

    return output_shape


def get_scale_from_size(input_size, output_size):
    """Get the scale factor given input size and output size.

    Args:
        input_size (tuple(int)): The size of the input image.
        output_size (tuple(int)): The size of the output image.

    Returns:
        list[float]: The scale factor of each dimension.
    """

    scale = [
        1.0 * output_shape / input_shape
        for (input_shape, output_shape) in zip(input_size, output_size)
    ]

    return scale


def _cubic(x):
    """Cubic function.

    Args:
        x (ndarray): The distance from the center position.

    Returns:
        ndarray: The weight corresponding to a particular distance.
    """

    x = np.array(x, dtype=np.float32)
    x_abs = np.abs(x)
    x_abs_sq = x_abs**2
    x_abs_cu = x_abs_sq * x_abs

    # if |x| <= 1: y = 1.5|x|^3 - 2.5|x|^2 + 1
    # if 1 < |x| <= 2: -0.5|x|^3 + 2.5|x|^2 - 4|x| + 2
    f = (1.5 * x_abs_cu - 2.5 * x_abs_sq + 1) * (x_abs <= 1) + (
        -0.5 * x_abs_cu + 2.5 * x_abs_sq - 4 * x_abs + 2) * ((1 < x_abs) &
                                                             (x_abs <= 2))

    return f


def get_weights_indices(input_length, output_length, scale, kernel,
                        kernel_width):
    """Get weights and indices for interpolation.

    Args:
        input_length (int): Length of the input sequence.
        output_length (int): Length of the output sequence.
        scale (float): Scale factor.
        kernel (func): The kernel used for resizing.
        kernel_width (int): The width of the kernel.

    Returns:
        list[ndarray]: The weights and the indices for interpolation.
    """
    if scale < 1:  # modified kernel for antialiasing

        def h(x):
            return scale * kernel(scale * x)

        kernel_width = 1.0 * kernel_width / scale
    else:
        h = kernel
        kernel_width = kernel_width

    # coordinates of output
    x = np.arange(1, output_length + 1).astype(np.float32)

    # coordinates of input
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)  # leftmost pixel
    p = int(np.ceil(kernel_width)) + 2  # maximum number of pixels

    # indices of input pixels
    ind = left[:, np.newaxis, ...] + np.arange(p)
    indices = ind.astype(np.int32)

    # weights of input pixels
    weights = h(u[:, np.newaxis, ...] - indices - 1)

    weights = weights / np.sum(weights, axis=1)[:, np.newaxis, ...]

    # remove all-zero columns
    aux = np.concatenate(
        (np.arange(input_length), np.arange(input_length - 1, -1,
                                            step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]

    return weights, indices


def resize_along_dim(img_in, weights, indices, dim):
    """Resize along a specific dimension.

    Args:
        img_in (ndarray): The input image.
        weights (ndarray): The weights used for interpolation, computed from
            [get_weights_indices].
        indices (ndarray): The indices used for interpolation, computed from
            [get_weights_indices].
        dim (int): Which dimension to undergo interpolation.

    Returns:
        ndarray: Interpolated (along one dimension) image.
    """

    img_in = img_in.astype(np.float32)
    w_shape = weights.shape
    output_shape = list(img_in.shape)
    output_shape[dim] = w_shape[0]
    img_out = np.zeros(output_shape)

    if dim == 0:
        for i in range(w_shape[0]):
            w = weights[i, :][np.newaxis, ...]
            ind = indices[i, :]
            img_slice = img_in[ind, :]
            img_out[i] = np.sum(np.squeeze(img_slice, axis=0) * w.T, axis=0)
    elif dim == 1:
        for i in range(w_shape[0]):
            w = weights[i, :][:, :, np.newaxis]
            ind = indices[i, :]
            img_slice = img_in[:, ind]
            img_out[:, i] = np.sum(np.squeeze(img_slice, axis=1) * w.T, axis=1)

    if img_in.dtype == np.uint8:
        img_out = np.clip(img_out, 0, 255)
        return np.around(img_out).astype(np.uint8)
    else:
        return img_out


@PIPELINES.register_module()
class MATLABLikeResize:
    """Resize the input image using MATLAB-like downsampling.

    Currently support bicubic interpolation only. Note that the output of
    this function is slightly different from the official MATLAB function.

    Required keys are the keys in attribute "keys". Added or modified keys
    are "scale" and "output_shape", and the keys in attribute "keys".

    Args:
        keys (list[str]): A list of keys whose values are modified.
        scale (float | None, optional): The scale factor of the resize
            operation. If None, it will be determined by output_shape.
            Default: None.
        output_shape (tuple(int) | None, optional): The size of the output
            image. If None, it will be determined by scale. Note that if
            scale is provided, output_shape will not be used.
            Default: None.
        kernel (str, optional): The kernel for the resize operation.
            Currently support 'bicubic' only. Default: 'bicubic'.
        kernel_width (float): The kernel width. Currently support 4.0 only.
            Default: 4.0.
    """

    def __init__(self,
                 keys,
                 scale=None,
                 output_shape=None,
                 kernel='bicubic',
                 kernel_width=4.0):

        if kernel.lower() != 'bicubic':
            raise ValueError('Currently support bicubic kernel only.')

        if float(kernel_width) != 4.0:
            raise ValueError('Current support only width=4 only.')

        if scale is None and output_shape is None:
            raise ValueError('"scale" and "output_shape" cannot be both None')

        self.kernel_func = _cubic
        self.keys = keys
        self.scale = scale
        self.output_shape = output_shape
        self.kernel = kernel
        self.kernel_width = kernel_width

    def _resize(self, img):
        weights = {}
        indices = {}

        # compute scale and output_size
        if self.scale is not None:
            scale = float(self.scale)
            scale = [scale, scale]
            output_size = get_size_from_scale(img.shape, scale)
        else:
            scale = get_scale_from_size(img.shape, self.output_shape)
            output_size = list(self.output_shape)

        # apply cubic interpolation along two dimensions
        order = np.argsort(np.array(scale))
        for k in range(2):
            key = (img.shape[k], output_size[k], scale[k], self.kernel_func,
                   self.kernel_width)
            weight, index = get_weights_indices(img.shape[k], output_size[k],
                                                scale[k], self.kernel_func,
                                                self.kernel_width)
            weights[key] = weight
            indices[key] = index

        output = np.copy(img)
        if output.ndim == 2:  # grayscale image
            output = output[:, :, np.newaxis]

        for k in range(2):
            dim = order[k]
            key = (img.shape[dim], output_size[dim], scale[dim],
                   self.kernel_func, self.kernel_width)
            output = resize_along_dim(output, weights[key], indices[key], dim)

        return output

    def __call__(self, results):
        for key in self.keys:
            is_single_image = False
            if isinstance(results[key], np.ndarray):
                is_single_image = True
                results[key] = [results[key]]

            results[key] = [self._resize(img) for img in results[key]]

            if is_single_image:
                results[key] = results[key][0]

        results['scale'] = self.scale
        results['output_shape'] = self.output_shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, scale={self.scale}, '
            f'output_shape={self.output_shape}, '
            f'kernel={self.kernel}, kernel_width={self.kernel_width})')
        return repr_str
