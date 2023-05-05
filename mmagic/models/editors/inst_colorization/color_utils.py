# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def xyz2rgb(xyz):
    """Conversion images from xyz to rgb.

    Args:
        xyz (tensor): The images to be conversion

    Returns:
        out (tensor): The converted image
    """
    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] \
        - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] \
        + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] \
        + 1.05731107 * xyz[:, 2, :, :]

    # sometimes reaches a small negative number, which causes NaNs
    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]),
                    dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()

    rgb = (1.055 * (rgb**(1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)
    return rgb


def lab2xyz(lab):
    """Conversion images from lab to xyz.

    Args:
        lab (tensor): The images to be conversion

    Returns:
        out (tensor): The converted image
    """
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)
    if (z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0, )).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0, )), z_int)

    out = torch.cat(
        (x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]),
        dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if (out.is_cuda):
        mask = mask.cuda()

    out = (out**3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out * sc
    return out


def lab2rgb(lab_rs, color_data_opt):
    """Conversion images from lab to rgb.

    Args:
        lab_rs (tensor): The images to be conversion
        color_data_opt (dict): Config for image colorspace transformation.
            Include: l_norm, ab_norm, l_cent

    Returns:
        out (tensor): The converted image
    """
    L = lab_rs[:,
               [0], :, :] * color_data_opt['l_norm'] + color_data_opt['l_cent']
    AB = lab_rs[:, 1:, :, :] * color_data_opt['ab_norm']
    lab = torch.cat((L, AB), dim=1)
    out = xyz2rgb(lab2xyz(lab))
    return out


def encode_ab_ind(data_ab, color_data_opt):
    """Encode ab value into an index.

    Args:
        data_ab: Nx2xHxW from [-1,1]
        color_data_opt: Config for image colorspace transformation.
            ab_max, ab_quant, ab_norm, ab_quant
    Returns:
            Nx1xHxW from [0,Q)
    """
    A = 2 * color_data_opt['ab_max'] / color_data_opt['ab_quant'] + 1
    data_ab_rs = torch.round(
        (data_ab * color_data_opt['ab_norm'] + color_data_opt['ab_max']) /
        color_data_opt['ab_quant'])  # normalized bin number
    data_q = data_ab_rs[:, [0], :, :] * A + data_ab_rs[:, [1], :, :]
    return data_q


def rgb2xyz(rgb):
    """Conversion images from rgb to xyz
    rgb from [0,1]
    xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]])
    Args:
        rgb (Tensor): image in rgb colorspace

    Returns:
         xyz (Tensor): image in xyz colorspace

    """
    mask = (rgb > .04045).type(torch.FloatTensor)
    if (rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb + .055) / 1.055)**2.4) * mask + rgb / 12.92 * (1 - mask)

    x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] \
        + .180423 * rgb[:, 2, :, :]
    y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] \
        + .072169 * rgb[:, 2, :, :]
    z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] \
        + .950227 * rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]),
                    dim=1)

    return out


def xyz2lab(xyz):
    """Conversion images from xyz to lab
    xyz from [0,1]
    factors: 0.95047, 1., 1.08883

    Args:
        xyz (Tensor): image in xyz colorspace

    Returns:
         out (Tensor): Image in lab colorspace
    """
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if (xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz / sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if (xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1 / 3.) * mask + (7.787 * xyz_scale +
                                            16. / 116.) * (1 - mask)

    L = 116. * xyz_int[:, 1, :, :] - 16.
    a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]),
                    dim=1)

    return out


def rgb2lab(rgb, color_opt):
    """Conversion images from rgb to lab.

    Args:
        data_raw (tensor): The images to be conversion
        color_opt (dict): Config for image colorspace transformation.
            Include: ab_thresh, ab_norm, sample_PS, mask_cent

    Returns:
        out (tensor): The converted image
    """
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:, [0], :, :] - color_opt['l_cent']) / color_opt['l_norm']
    ab_rs = lab[:, 1:, :, :] / color_opt['ab_norm']
    out = torch.cat((l_rs, ab_rs), dim=1)
    return out


def get_colorization_data(data_raw, color_opt, num_points=None):
    """Conversion images from rgb to lab.

    Args:
        data_raw (tensor): The images to be conversion
        color_opt (dict): Config for image colorspace transformation.
            Include: ab_thresh, ab_norm, sample_PS, mask_cent

    Returns:
        results (dict): Output in add_color_patches_rand_gt
    """
    data = {}
    data_lab = rgb2lab(data_raw, color_opt)
    data['A'] = data_lab[:, [
        0,
    ], :, :]
    data['B'] = data_lab[:, 1:, :, :]

    # mask out grayscale images
    if color_opt['ab_thresh'] > 0:
        thresh = 1. * color_opt['ab_thresh'] / color_opt['ab_norm']
        mask = torch.sum(
            torch.abs(
                torch.max(torch.max(data['B'], dim=3)[0], dim=2)[0] -
                torch.min(torch.min(data['B'], dim=3)[0], dim=2)[0]),
            dim=1) >= thresh
        data['A'] = data['A'][mask, :, :, :]
        data['B'] = data['B'][mask, :, :, :]
        if torch.sum(mask) == 0:
            return None

    return add_color_patches_rand_gt(
        data, color_opt, p=color_opt['p'], num_points=num_points)


def add_color_patches_rand_gt(data,
                              color_opt,
                              p=.125,
                              num_points=None,
                              use_avg=True,
                              samp='normal'):
    """Add random color points sampled from ground truth based on: Number of
    points.

    - if num_points is 0, then sample from geometric distribution,
        drawn from probability p
    - if num_points > 0, then sample that number of points
    Location of points
    - if samp is 'normal', draw from N(0.5, 0.25) of image
    - otherwise, draw from U[0, 1] of image

    Args:
        data (tensor): The images to be conversion
        color_opt (dict): Config for image colorspace transformation
            Include: ab_thresh, ab_norm, sample_PS, mask_cent
        p (float): Sampling geometric distribution, 1.0 means no hints
        num_points (int): Certain number of points
        use_avg (bool): Whether to use the mean when add color point
            Default: True.
        samp (str): Geometric distribution or uniform distribution when
            sample location. Default: normal.

    Returns:
        results (dict): Result dict from :obj:``mmcv.BaseDataset``.
    """
    N, C, H, W = data['B'].shape

    data['hint_B'] = torch.zeros_like(data['B'])
    data['mask_B'] = torch.zeros_like(data['A'])

    for nn in range(N):
        pp = 0
        cont_cond = True
        while cont_cond:
            # draw from geometric
            if num_points is None:
                cont_cond = np.random.rand() < (1 - p)
            else:
                # add certain number of points
                cont_cond = pp < num_points
            # skip out of loop if condition not met
            if not cont_cond:
                continue

            # patch size
            P = np.random.choice(color_opt['sample_PS'])
            # sample location: geometric distribution
            if samp == 'normal':
                h = int(
                    np.clip(
                        np.random.normal((H - P + 1) / 2., (H - P + 1) / 4.),
                        0, H - P))
                w = int(
                    np.clip(
                        np.random.normal((W - P + 1) / 2., (W - P + 1) / 4.),
                        0, W - P))
            else:  # uniform distribution
                h = np.random.randint(H - P + 1)
                w = np.random.randint(W - P + 1)

            # add color point
            if use_avg:
                data['hint_B'][nn, :, h:h + P, w:w + P] = torch.mean(
                    torch.mean(
                        data['B'][nn, :, h:h + P, w:w + P],
                        dim=2,
                        keepdim=True),
                    dim=1,
                    keepdim=True).view(1, C, 1, 1)
            else:
                data['hint_B'][nn, :, h:h + P, w:w + P] = \
                    data['B'][nn, :, h:h + P, w:w + P]

            data['mask_B'][nn, :, h:h + P, w:w + P] = 1

            # increment counter
            pp += 1

    data['mask_B'] -= color_opt['mask_cent']

    return data
