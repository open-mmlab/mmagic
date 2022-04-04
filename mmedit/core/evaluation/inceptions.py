# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models.inception as inception
from mmcv.runner import load_checkpoint
from scipy import linalg

from mmedit.utils import get_root_logger
from ..registry import METRICS


def img2tensor(img, out_type=torch.float32):
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img / 255.
    tensor = torch.from_numpy(img)
    return tensor.to(out_type)


class InceptionV3:
    """Inception network used in calculating perceptual loss.
    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of inception network. Note that the
    pretrained path must fit the inception type.
    Args:
        inception_type (str): Set the type of inception network.
            Default: 'inception_v3'.
        use_input_norm (bool): If True, normalize the input image.
            Importantly, the input feature must in the range [0, 1].
            Default: True.
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://inception_v3_google'
    """

    def __init__(self,
                 inception_type='inception_v3',
                 use_input_norm=True,
                 pretrained='torchvision://inception_v3_google'):
        super().__init__()
        if pretrained.startswith('torchvision://'):
            assert inception_type in pretrained
        self.use_input_norm = use_input_norm

        # get inception model and load pretrained inception weight
        # remove _inception from attributes to avoid `find_unused_parameters`
        _inception = getattr(inception, inception_type)()
        self.init_weights(_inception, pretrained)
        del _inception.AuxLogits, _inception.dropout, _inception.fc
        self.inception = _inception.eval()

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            # the std is for image with range [-1, 1]
            self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        for v in self.inception.parameters():
            v.requires_grad = False

    def __call__(self, img1, img2, crop_border=0):
        return self.forward(img1), self.forward(img2)

    def forward(self, x):
        """Forward function.
        Args:
            x (np.ndarray): Input np.ndarray with shape (h, w, c).
        Returns:
            np.ndarray: Forward results, which is feature np.ndarray
                with shape (1, 2048).
        """

        x = img2tensor(x)
        x = F.interpolate(
            x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for _, module in self.inception.named_children():
            x = module(x)
        output = torch.flatten(x, 1)
        return output.numpy()

    def init_weights(self, model, pretrained):
        """Init weights.
        Args:
            model (nn.Module): Models to be inited.
            pretrained (str): Path for pretrained weights.
        """
        logger = get_root_logger()
        load_checkpoint(model, pretrained, logger=logger)


def frechet_distance(x, y):
    """Numpy implementation of the Frechet Distance."""

    mu1, sigma1 = np.mean(x, axis=0), np.cov(x, rowvar=False)
    mu2, sigma2 = np.mean(y, axis=0), np.cov(y, rowvar=False)
    mu1, sigma1 = np.atleast_1d(mu1), np.atleast_2d(sigma1) + 1e-7
    mu2, sigma2 = np.atleast_1d(mu2), np.atleast_2d(sigma2) + 1e-7

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    frechet_distance = diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * np.trace(covmean)
    return frechet_distance


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef=1):
    if Y is None:
        Y = X.copy()
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (np.matmul(X, Y.T) * gamma + coef)**degree
    return K


def mmd2(x, y):
    """Numpy implementation of the Maximum Mean Discrepancy."""

    XX = polynomial_kernel(x, degree=3, coef0=1)
    YY = polynomial_kernel(y, degree=3, coef0=1)
    XY = polynomial_kernel(x, y, degree=3, coef0=1)

    m = XX.shape[0]
    mmd2 = 1 / (m * (m - 1)) * (np.sum(XX) - np.trace(XX)) + (
        1 / (m *
             (m - 1)) * (np.sum(YY) - np.trace(YY))) - (2 / m**2 * np.sum(XY))
    return mmd2


@METRICS.register_module()
class FID:
    """FID metric.

    Args:
        num_bootstraps (int): Number of repetitions. Default: 100.
    """

    def __init__(self, num_bootstraps=1):
        self.num_bootstraps = num_bootstraps

    def __call__(self, X, Y):
        """Calculate FID.

        Args:
            X (np.ndarray): Input feature X with shape (n_samples, dims).
            Y (np.ndarray): Input feature Y with shape (n_samples, dims).

        Returns:
            (float): fid value.
        """
        num_samples = X.shape[0]
        fids = np.zeros(self.num_bootstraps)
        for i in range(self.num_bootstraps):
            fids[i] = frechet_distance(
                X[np.random.choice(num_samples, num_samples, replace=True)],
                Y[np.random.choice(num_samples, num_samples, replace=True)],
            )
        return fids.mean()


@METRICS.register_module()
class KID:
    """KID metric.

    Args:
        num_bootstraps (int): Number of repetitions. Default: 100.
        sample_size (int): Size to sample. Default: 1000.
    """

    def __init__(self, num_bootstraps=100, sample_size=1000):
        self.num_bootstraps = num_bootstraps
        self.sample_size = sample_size

    def __call__(self, X, Y):
        """Calculate KID.

        Args:
            X (np.ndarray): Input feature X with shape (n_samples, dims).
            Y (np.ndarray): Input feature Y with shape (n_samples, dims).

        Returns:
            (Tuple[float, float]): Tuple of mean and std of kid value.
        """
        num_samples = X.shape[0]
        kids = np.zeros(self.num_bootstraps)
        for i in range(self.num_bootstraps):
            kids[i] = mmd2(
                X[np.random.choice(
                    num_samples, self.sample_size, replace=False)],
                Y[np.random.choice(
                    num_samples, self.sample_size, replace=False)],
            )
        return kids.mean(), kids.std()
