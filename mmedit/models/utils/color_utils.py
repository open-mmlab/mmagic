# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


# Color conversion code
def rgb2xyz(rgb):  # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
    # [0.212671, 0.715160, 0.072169],
    # [0.019334, 0.119193, 0.950227]])

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

    # if(torch.sum(torch.isnan(out))>0):
    # print('rgb2xyz')
    # embed()
    return out


def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] \
        - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] \
        + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] \
        + 1.05731107 * xyz[:, 2, :, :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]),
                    dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))
    # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()

    rgb = (1.055 * (rgb**(1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)
    # if(torch.sum(torch.isnan(rgb))>0):
    # print('xyz2rgb')
    # embed()
    return rgb


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
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

    # if(torch.sum(torch.isnan(out))>0):
    # print('xyz2lab')
    # embed()

    return out


def lab2xyz(lab):
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

    # if(torch.sum(torch.isnan(out))>0):
    # print('lab2xyz')
    # embed()

    return out


def rgb2lab(rgb, **kwargs):
    lab = xyz2lab(rgb2xyz(rgb))
    # print(lab[0, 0, 0, 0])
    # lab_0 = lab[:, [0], :, :]
    l_rs = (lab[:, [0], :, :] - kwargs['l_cent']) / kwargs['l_norm']
    # print(l_rs[0, 0, 0, 0])
    ab_rs = lab[:, 1:, :, :] / kwargs['ab_norm']
    out = torch.cat((l_rs, ab_rs), dim=1)
    # if(torch.sum(torch.isnan(out))>0):
    # print('rgb2lab')
    # embed()
    return out


def lab2rgb(lab_rs, **kwargs):
    L = lab_rs[:, [0], :, :] * kwargs['l_norm'] + kwargs['l_cent']
    AB = lab_rs[:, 1:, :, :] * kwargs['ab_norm']
    lab = torch.cat((L, AB), dim=1)
    out = xyz2rgb(lab2xyz(lab))
    # if(torch.sum(torch.isnan(out))>0):
    # print('lab2rgb')
    # embed()
    return out


def get_colorization_data(data_raw,
                          ab_thresh=5.,
                          p=.125,
                          num_points=None,
                          **kwargs):
    data = {}

    data_lab = rgb2lab(data_raw, **kwargs)
    data['A'] = data_lab[:, [
        0,
    ], :, :]
    data['B'] = data_lab[:, 1:, :, :]

    if ab_thresh > 0:  # mask out grayscale images
        thresh = 1. * ab_thresh / kwargs['ab_norm']
        mask = torch.sum(
            torch.abs(
                torch.max(torch.max(data['B'], dim=3)[0], dim=2)[0] -
                torch.min(torch.min(data['B'], dim=3)[0], dim=2)[0]),
            dim=1) >= thresh
        data['A'] = data['A'][mask, :, :, :]
        data['B'] = data['B'][mask, :, :, :]
        # print('Removed %i points'%torch.sum(mask==0).numpy())
        if torch.sum(mask) == 0:
            return None

    return add_color_patches_rand_gt(
        data, p=p, num_points=num_points, **kwargs)


def add_color_patches_rand_gt(data,
                              p=.125,
                              num_points=None,
                              use_avg=True,
                              samp='normal',
                              **kwargs):
    # Add random color points sampled from ground truth based on:
    #   Number of points
    #   - if num_points is 0, then sample from geometric distribution,
    #       drawn from probability p
    #   - if num_points > 0, then sample that number of points
    #   Location of points
    #   - if samp is 'normal', draw from N(0.5, 0.25) of image
    #   - otherwise, draw from U[0, 1] of image
    N, C, H, W = data['B'].shape

    data['hint_B'] = torch.zeros_like(data['B'])
    data['mask_B'] = torch.zeros_like(data['A'])

    for nn in range(N):
        pp = 0
        cont_cond = True
        while cont_cond:
            if num_points is None:  # draw from geometric
                # embed()
                cont_cond = np.random.rand() < (1 - p)
            else:  # add certain number of points
                cont_cond = pp < num_points
            if not cont_cond:  # skip out of loop if condition not met
                continue

            P = np.random.choice(kwargs['sample_PS'])  # patch size

            # sample location
            if samp == 'normal':  # geometric distribution
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
                # embed()
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

    data['mask_B'] -= kwargs['mask_cent']

    return data


def encode_ab_ind(data_ab, **kwargs):
    # Encode ab value into an index
    # INPUTS
    #   data_ab   Nx2xHxW \in [-1,1]
    # OUTPUTS
    #   data_q    Nx1xHxW \in [0,Q)
    A = 2 * kwargs['ab_max'] / kwargs['ab_quant'] + 1
    data_ab_rs = torch.round((data_ab * kwargs['ab_norm'] + kwargs['ab_max']) /
                             kwargs['ab_quant'])  # normalized bin number
    data_q = data_ab_rs[:, [0], :, :] * A + data_ab_rs[:, [1], :, :]
    return data_q
