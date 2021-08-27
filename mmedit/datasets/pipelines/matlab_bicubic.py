from math import ceil

import numpy as np


def get_size_from_scale(input_size, scale_factor):
    """Get the output size given input size and scale factor.

    Args:
        input_size (tuple): The size of the input image.
        scale_factor (float): The resize factor.

    Returns:
        list[int]: The size of the output image.
    """

    output_shape = [
        int(ceil(scale * shape))
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


def cubic(x):
    """ Cubic function.

    Args:
        x (ndarray): The distance from the center position.

    Returns:
        ndarray: The weight corresponding to a particular distance.

    """

    x = np.array(x).astype(np.float64)
    x_abs = np.absolute(x)
    x_abs_sq = np.multiply(x_abs, x_abs)
    x_abs_cu = np.multiply(x_abs_sq, x_abs)
    f = np.multiply(1.5 * x_abs_cu - 2.5 * x_abs_sq + 1,
                    x_abs <= 1) + np.multiply(
                        -0.5 * x_abs_cu + 2.5 * x_abs_sq - 4 * x_abs + 2,
                        (1 < x_abs) & (x_abs <= 2))

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
        list[ndarry]: The weights and the indices for interpolation.


    """
    if scale < 1:  # modified kernel for antialiasing

        def h(x):
            return scale * kernel(scale * x)

        kernel_width = 1.0 * kernel_width / scale
    else:
        h = kernel
        kernel_width = kernel_width

    # coordinates of output
    x = np.arange(1, output_length + 1).astype(np.float64)

    # coordinates of input
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)  # leftmost pixel
    p = int(ceil(kernel_width)) + 2  # maximum number of pixels

    # indices of input pixels
    ind = np.expand_dims(left, axis=1) + np.arange(p) - 1
    indices = ind.astype(np.int32)

    # weights of input pixels
    weights = h(np.expand_dims(u, axis=1) - indices - 1)
    weights = np.divide(weights,
                        np.expand_dims(np.sum(weights, axis=1), axis=1))

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

    img_in = img_in.astype(np.float64)
    w_shape = weights.shape
    output_shape = list(img_in.shape)
    output_shape[dim] = w_shape[0]
    img_out = np.zeros(output_shape)

    if dim == 0:
        for i in range(w_shape[0]):
            w = np.expand_dims(weights[i, :], 0)
            ind = indices[i, :]
            img_slice = img_in[ind, :]
            img_out[i] = np.sum(
                np.multiply(np.squeeze(img_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i in range(w_shape[0]):
            w = np.expand_dims(weights[i, :], 2)
            ind = indices[i, :]
            img_slice = img_in[:, ind]
            img_out[:, i] = np.sum(
                np.multiply(np.squeeze(img_slice, axis=1), w.T), axis=1)

    if img_in.dtype == np.uint8:
        img_out = np.clip(img_out, 0, 255)
        return np.around(img_out).astype(np.uint8)
    else:
        return img_out


def imresize(img,
             scale=None,
             output_shape=None,
             kernel='cubic',
             kernel_width=4.0):
    """Resize the input image using MATLAB downsampling.

    Currently support bicubic interpolation only. Note that the output of this
    function is slightly different from the official MATLAB function.

    Args:
        img (ndarray): The input image to be resized.
        scale (float | None, optional): The scale factor of the resize
            operation. If None, it will be determined by output_shape.
            Default: None.
        output_shape (tuple(int) | None, optional): The size of the output
            image. If None, it will be determined by scale. Note that if scale
            is provided, output_shape will not be used. Default: None.
        kernel (str, optional): The kernel for the resize operation. Currently
            support 'cubic' only. Default: 'cubic'.
        kernel_width (float): The kernel width. Currently support 4.0 only.
            Default: 4.0.

    Returns:
        ndarray: The output image.
    """

    if kernel.lower() != 'cubic':
        raise ValueError('Currently support bicubic kernel only.')
    else:
        kernel = cubic

    if float(kernel_width) != 4.0:
        raise ValueError('Current support only width=4 only.')

    weights = {}
    indices = {}

    # compute scale and output_size
    if scale is not None:
        scale = float(scale)
        scale = [scale, scale]
        output_size = get_size_from_scale(img.shape, scale)
    elif output_shape is not None:
        scale = get_scale_from_size(img.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('"scale" and "output_shape" cannot be both None')

    # apply cubic interpolation along two dimensions
    order = np.argsort(np.array(scale))
    for k in range(2):
        key = (img.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weight, index = get_weights_indices(img.shape[k], output_size[k],
                                            scale[k], kernel, kernel_width)
        weights[key] = weight
        indices[key] = index

    output = np.copy(img)
    if output.ndim == 2:  # grayscale image
        output = np.expand_dims(output, axis=2)

    for k in range(2):
        dim = order[k]
        key = (img.shape[dim], output_size[dim], scale[dim], kernel,
               kernel_width)
        output = resize_along_dim(output, weights[key], indices[key], dim)

    return output.squeeze(2)
