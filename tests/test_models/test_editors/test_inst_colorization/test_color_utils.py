# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmagic.models.editors.inst_colorization import color_utils


class TestColorUtils:
    color_data_opt = dict(
        ab_thresh=0,
        p=1.0,
        sample_PS=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ],
        ab_norm=110,
        ab_max=110.,
        ab_quant=10.,
        l_norm=100.,
        l_cent=50.,
        mask_cent=0.5)

    def test_xyz2lab(self):
        xyz = torch.rand(1, 3, 8, 8)
        lab = color_utils.xyz2lab(xyz)

        sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
        xyz_scale = xyz / sc
        mask = (xyz_scale > .008856).type(torch.FloatTensor)

        xyz_int = xyz_scale**(1 / 3.) * mask + (7.787 * xyz_scale +
                                                16. / 116.) * (1 - mask)
        L = 116. * xyz_int[:, 1, :, :] - 16.
        a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
        b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])

        assert lab.shape == (1, 3, 8, 8)
        assert lab.equal(
            torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]),
                      dim=1))

    def test_rgb2xyz(self):
        rgb = torch.rand(1, 3, 8, 8)
        xyz = color_utils.rgb2xyz(rgb)

        mask = (rgb > .04045).type(torch.FloatTensor)
        rgb = (((rgb + .055) / 1.055)**2.4) * mask + rgb / 12.92 * (1 - mask)

        x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] \
            + .180423 * rgb[:, 2, :, :]
        y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] \
            + .072169 * rgb[:, 2, :, :]
        z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] \
            + .950227 * rgb[:, 2, :, :]

        assert xyz.shape == (1, 3, 8, 8)
        assert xyz.equal(
            torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]),
                      dim=1))

    def test_rgb2lab(self):
        rgb = torch.rand(1, 3, 8, 8)
        lab = color_utils.rgb2lab(rgb, self.color_data_opt)
        _lab = color_utils.xyz2lab(color_utils.rgb2xyz(rgb))

        l_rs = (_lab[:, [0], :, :] -
                self.color_data_opt['l_cent']) / self.color_data_opt['l_norm']
        ab_rs = _lab[:, 1:, :, :] / self.color_data_opt['ab_norm']

        assert lab.shape == (1, 3, 8, 8)
        assert lab.equal(torch.cat((l_rs, ab_rs), dim=1))

    def test_lab2rgb(self):
        lab = torch.rand(1, 3, 8, 8)
        rgb = color_utils.lab2rgb(lab, self.color_data_opt)

        L = lab[:, [0], :, :] * self.color_data_opt[
            'l_norm'] + self.color_data_opt['l_cent']
        AB = lab[:, 1:, :, :] * self.color_data_opt['ab_norm']

        lab = torch.cat((L, AB), dim=1)

        assert rgb.shape == (1, 3, 8, 8)
        assert rgb.equal(color_utils.xyz2rgb(color_utils.lab2xyz(lab)))

    def test_lab2xyz(self):
        lab = torch.rand(1, 3, 8, 8)
        xyz = color_utils.lab2xyz(lab)
        y_int = (lab[:, 0, :, :] + 16.) / 116.
        x_int = (lab[:, 1, :, :] / 500.) + y_int
        z_int = y_int - (lab[:, 2, :, :] / 200.)
        z_int = torch.max(torch.Tensor((0, )), z_int)

        out = torch.cat(
            (x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]),
            dim=1)
        mask = (out > .2068966).type(torch.FloatTensor)
        sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
        out = (out**3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)
        target = sc * out
        assert xyz.shape == (1, 3, 8, 8)
        assert xyz.equal(target)

    def test_xyz2rgb(self):
        xyz = torch.rand(1, 3, 8, 8)

        rgb = color_utils.xyz2rgb(xyz)

        r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] \
            - 0.49853633 * xyz[:, 2, :, :]
        g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] \
            + .04155593 * xyz[:, 2, :, :]
        b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] \
            + 1.05731107 * xyz[:, 2, :, :]

        _rgb = torch.cat(
            (r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
        _rgb = torch.max(_rgb, torch.zeros_like(_rgb))

        mask = (_rgb > .0031308).type(torch.FloatTensor)

        assert rgb.shape == (1, 3, 8, 8) and mask.shape == (1, 3, 8, 8)
        assert rgb.equal((1.055 * (_rgb**(1. / 2.4)) - 0.055) * mask +
                         12.92 * _rgb * (1 - mask))

    def test_get_colorization_data(self):
        data_raw = torch.rand(1, 3, 8, 8)

        res = color_utils.get_colorization_data(data_raw, self.color_data_opt)

        assert isinstance(res, dict)
        assert 'A' in res.keys() and 'B' in res.keys() \
               and 'hint_B' in res.keys() and 'mask_B' in res.keys()
        assert res['A'].shape == res['mask_B'].shape == (1, 1, 8, 8)
        assert res['hint_B'].shape == res['B'].shape == (1, 2, 8, 8)

    def test_encode_ab_ind(self):
        data_ab = torch.rand(1, 2, 8, 8)
        data_q = color_utils.encode_ab_ind(data_ab, self.color_data_opt)
        A = 2 * 110. / 10. + 1

        data_ab_rs = torch.round((data_ab * 110 + 110.) / 10.)

        assert data_q.shape == (1, 1, 8, 8)
        assert data_q.equal(data_ab_rs[:, [0], :, :] * A +
                            data_ab_rs[:, [1], :, :])


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
