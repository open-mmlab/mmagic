import torch
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module()
class LapLoss(torch.nn.Module):

    def __init__(self,
                 max_levels=5,
                 channels=1,
                 device=torch.device('cuda'),
                 loss_weight=1.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = LapLoss._gauss_kernel(
            channels=channels, device=device)
        self.loss_weight = loss_weight

    def forward(self, input, target, weight=None):
        pyr_input = LapLoss._laplacian_pyramid(
            img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = LapLoss._laplacian_pyramid(
            img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        loss_lap = sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        return loss_lap * self.loss_weight

    @staticmethod
    def _gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1], [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.], [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    @staticmethod
    def _downsample(x):
        return x[:, :, ::2, ::2]

    @staticmethod
    def _upsample(x):
        cc = torch.cat([
            x,
            torch.zeros(
                x.shape[0],
                x.shape[1],
                x.shape[2],
                x.shape[3],
                device=x.device)
        ],
                       dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)

        cc = torch.cat([
            cc,
            torch.zeros(
                x.shape[0],
                x.shape[1],
                x.shape[3],
                x.shape[2] * 2,
                device=x.device)
        ],
                       dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return LapLoss._conv_gauss(
            x_up,
            4 * LapLoss._gauss_kernel(channels=x.shape[1], device=x.device))

    @staticmethod
    def _conv_gauss(img, kernel):
        img = F.pad(img, (2, 2, 2, 2), mode='reflect')
        out = F.conv2d(img, kernel, groups=img.shape[1])
        return out

    @staticmethod
    def _laplacian_pyramid(img, kernel, max_levels=3):
        current = img
        pyr = []
        for level in range(max_levels):
            filtered = LapLoss._conv_gauss(current, kernel)
            down = LapLoss._downsample(filtered)
            up = LapLoss._upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

    @staticmethod
    def _weight_pyramid(img, max_levels=3):
        current = img
        pyr = []
        for level in range(max_levels):
            down = LapLoss._downsample(current)
            pyr.append(current)
            current = down
        return pyr
