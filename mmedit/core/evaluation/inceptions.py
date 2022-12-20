# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from scipy import linalg

from ..registry import METRICS
from .inception_utils import load_inception


class InceptionV3:
    """Feature extractor features using InceptionV3 model.

    Args:
        style (str): The model style to run Inception model. it must be either
            'StyleGAN' or 'pytorch'.
        device (torch.device): device to extract feature.
        inception_kwargs (**kwargs): kwargs for InceptionV3.
    """

    def __init__(self, style='StyleGAN', device='cpu', **inception_kwargs):
        self.inception = load_inception(
            style=style, **inception_kwargs).eval().to(device)
        self.style = style
        self.device = device

    def __call__(self, img1, img2, crop_border=0):
        """Extract features of real and fake images.

        Args:
            img1, img2 (np.ndarray): Images with range [0, 255]
                and shape (H, W, C).

        Returns:
            (tuple): Pair of features extracted from InceptionV3 model.
        """
        return (
            self.forward_inception(self.img2tensor(img1)).numpy(),
            self.forward_inception(self.img2tensor(img2)).numpy(),
        )

    def img2tensor(self, img):
        img = np.expand_dims(img.transpose((2, 0, 1)), axis=0)
        if self.style == 'StyleGAN':
            return torch.tensor(img).to(device=self.device, dtype=torch.uint8)

        return torch.from_numpy(img / 255.).to(
            device=self.device, dtype=torch.float32)

    def forward_inception(self, x):
        if self.style == 'StyleGAN':
            return self.inception(x).cpu()

        return self.inception(x)[-1].view(x.shape[0], -1).cpu()


def frechet_distance(X, Y):
    """Compute the frechet distance."""

    muX, covX = np.mean(X, axis=0), np.cov(X, rowvar=False)
    muY, covY = np.mean(Y, axis=0), np.cov(Y, rowvar=False)

    cov_sqrt = linalg.sqrtm(covX.dot(covY))
    frechet_distance = np.square(muX - muY).sum() + np.trace(covX) + np.trace(
        covY) - 2 * np.trace(cov_sqrt)
    return np.real(frechet_distance)


@METRICS.register_module()
class FID:
    """FID metric."""

    def __call__(self, X, Y):
        """Calculate FID.

        Args:
            X (np.ndarray): Input feature X with shape (n_samples, dims).
            Y (np.ndarray): Input feature Y with shape (n_samples, dims).

        Returns:
            (float): fid value.
        """
        return frechet_distance(X, Y)


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef=1):
    """Create a polynomial kernel."""
    Y = X if Y is None else Y
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = ((X @ Y.T) * gamma + coef)**degree
    return K


def mmd2(X, Y, unbiased=True):
    """Compute the Maximum Mean Discrepancy."""
    XX = polynomial_kernel(X, X)
    YY = polynomial_kernel(Y, Y)
    XY = polynomial_kernel(X, Y)

    m = X.shape[0]
    if not unbiased:
        return (np.sum(XX) + np.sum(YY) - 2 * np.sum(XY)) / m**2

    trX = np.trace(XX)
    trY = np.trace(YY)
    return (np.sum(XX) - trX + np.sum(YY) -
            trY) / (m * (m - 1)) - 2 * np.sum(XY) / m**2


@METRICS.register_module()
class KID:
    """Implementation of `KID <https://arxiv.org/abs/1801.01401>`.

    Args:
        num_repeats (int): The number of repetitions. Default: 100.
        sample_size (int): Size to sample. Default: 1000.
        use_unbiased_estimator (bool): Whether to use KID as an unbiased
            estimator. Using an unbiased estimator is desirable in the case of
            finite sample size, especially when the number of samples are
            small. Using an unbiased estimator is recommended in most cases.
            Default: True
    """

    def __init__(self,
                 num_repeats=100,
                 sample_size=1000,
                 use_unbiased_estimator=True):
        self.num_repeats = num_repeats
        self.sample_size = sample_size
        self.unbiased = use_unbiased_estimator

    def __call__(self, X, Y):
        """Calculate KID.

        Args:
            X (np.ndarray): Input feature X with shape (n_samples, dims).
            Y (np.ndarray): Input feature Y with shape (n_samples, dims).

        Returns:
            (dict): dict containing mean and std of KID values.
        """
        num_samples = X.shape[0]
        kid = list()
        for i in range(self.num_repeats):
            X_ = X[np.random.choice(
                num_samples, self.sample_size, replace=False)]
            Y_ = Y[np.random.choice(
                num_samples, self.sample_size, replace=False)]
            kid.append(mmd2(X_, Y_, unbiased=self.unbiased))
        kid = np.array(kid)
        return dict(KID_MEAN=kid.mean(), KID_STD=kid.std())
