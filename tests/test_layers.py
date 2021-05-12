import torch

from mmedit.models.components.layers import MeanShift


def test_mean_shift():
    rgb_mean = (1, 2, 3)
    rgb_std = (1, 0.5, 0.25)
    mean_shift = MeanShift(1, rgb_mean, rgb_std)
    x = torch.randn((2, 3, 64, 64))
    y = mean_shift(x)
    x = x.permute((1, 0, 2, 3)).reshape((3, -1))
    y = y.permute((1, 0, 2, 3)).reshape((3, -1))
    rgb_mean = torch.tensor(rgb_mean)
    rgb_std = torch.tensor(rgb_std)
    mean_x = x.mean(dim=1)
    mean_y = y.mean(dim=1)
    std_x = x.std(dim=1)
    std_y = y.std(dim=1)
    assert sum(torch.div(std_x, std_y) - rgb_std) < 1e-5
    assert sum(torch.div(mean_x - rgb_mean, rgb_std) - mean_y) < 1e-5


if __name__ == '__main__':
    test_mean_shift()
