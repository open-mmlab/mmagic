# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d

from ..registry import LOSSES


@LOSSES.register_module()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight
        if self.gan_type == 'smgan':
            self.gaussian_blur = GaussianBlur()

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan' or self.gan_type == 'smgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """

        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type == 'wgan':
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False, mask=None):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the target is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """

        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        elif self.gan_type == 'smgan':

            input_height, input_width = input.shape[2:]
            mask_height, mask_width = mask.shape[2:]

            # Handle inconsistent size between outputs and masks
            if input_height != mask_height or input_width != mask_width:
                input = F.interpolate(
                    input,
                    size=(mask_height, mask_width),
                    mode='bilinear',
                    align_corners=True)

                target_label = self.get_target_label(input, target_is_real)

            if is_disc:
                if target_is_real:
                    target_label = target_label
                else:
                    target_label = self.gaussian_blur(mask).detach().cuda(
                    ) if mask.is_cuda else self.gaussian_blur(
                        mask).detach().cpu()
                    # target_label = self.gaussian_blur(mask).detach().cpu()
                loss = self.loss(input, target_label)
            else:
                loss = self.loss(input, target_label) * mask / mask.mean()
                loss = loss.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSSES.register_module()
class GaussianBlur(nn.Module):
    """A Gaussian filter which blurs a given tensor with a two-dimensional
    gaussian kernel by convolving it along each channel. Batch operation is
    supported.

    This function is modified from kornia.filters.gaussian:
    `<https://kornia.readthedocs.io/en/latest/_modules/kornia/filters/gaussian.html>`.

    Args:
        kernel_size (tuple[int]): The size of the kernel. Default: (71, 71).
        sigma (tuple[float]): The standard deviation of the kernel.
        Default (10.0, 10.0)

    Returns:
        Tensor: The Gaussian-blurred tensor.

    Shape:
        - input: Tensor with shape of (n, c, h, w)
        - output: Tensor with shape of (n, c, h, w)
    """

    def __init__(self, kernel_size=(71, 71), sigma=(10.0, 10.0)):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = self.compute_zero_padding(kernel_size)
        self.kernel = self.get_2d_gaussian_kernel(kernel_size, sigma)

    @staticmethod
    def compute_zero_padding(kernel_size):
        """Compute zero padding tuple."""

        padding = [(ks - 1) // 2 for ks in kernel_size]

        return padding[0], padding[1]

    def get_2d_gaussian_kernel(self, kernel_size, sigma):
        """Get the two-dimensional Gaussian filter matrix coefficients.

        Args:
            kernel_size (tuple[int]): Kernel filter size in the x and y
                                      direction. The kernel sizes
                                      should be odd and positive.
            sigma (tuple[int]): Gaussian standard deviation in
                                the x and y direction.

        Returns:
            kernel_2d (Tensor): A 2D torch tensor with gaussian filter
                                matrix coefficients.
        """

        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise TypeError(
                'kernel_size must be a tuple of length two. Got {}'.format(
                    kernel_size))
        if not isinstance(sigma, tuple) or len(sigma) != 2:
            raise TypeError(
                'sigma must be a tuple of length two. Got {}'.format(sigma))

        kernel_size_x, kernel_size_y = kernel_size
        sigma_x, sigma_y = sigma

        kernel_x = self.get_1d_gaussian_kernel(kernel_size_x, sigma_x)
        kernel_y = self.get_1d_gaussian_kernel(kernel_size_y, sigma_y)
        kernel_2d = torch.matmul(
            kernel_x.unsqueeze(-1),
            kernel_y.unsqueeze(-1).t())

        return kernel_2d

    def get_1d_gaussian_kernel(self, kernel_size, sigma):
        """Get the Gaussian filter coefficients in one dimension (x or y
        direction).

        Args:
            kernel_size (int): Kernel filter size in x or y direction.
                               Should be odd and positive.
            sigma (float): Gaussian standard deviation in x or y direction.

        Returns:
            kernel_1d (Tensor): A 1D torch tensor with gaussian filter
                                coefficients in x or y direction.
        """

        if not isinstance(kernel_size,
                          int) or kernel_size % 2 == 0 or kernel_size <= 0:
            raise TypeError(
                'kernel_size must be an odd positive integer. Got {}'.format(
                    kernel_size))

        kernel_1d = self.gaussian(kernel_size, sigma)
        return kernel_1d

    def gaussian(self, kernel_size, sigma):

        def gauss_arg(x):
            return -(x - kernel_size // 2)**2 / float(2 * sigma**2)

        gauss = torch.stack([
            torch.exp(torch.tensor(gauss_arg(x))) for x in range(kernel_size)
        ])
        return gauss / gauss.sum()

    def forward(self, x):
        if not torch.is_tensor(x):
            raise TypeError(
                'Input x type is not a torch.Tensor. Got {}'.format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError(
                'Invalid input shape, we expect BxCxHxW. Got: {}'.format(
                    x.shape))
        _, c, _, _ = x.shape
        tmp_kernel = self.kernel.to(x.device).to(x.dtype)
        kernel = tmp_kernel.repeat(c, 1, 1, 1)

        return conv2d(x, kernel, padding=self.padding, stride=1, groups=c)


def gradient_penalty_loss(discriminator, real_data, fake_data, mask=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        mask (Tensor): Masks for inpainting. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(real_data)

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if mask is not None:
        gradients = gradients * mask

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if mask is not None:
        gradients_penalty /= torch.mean(mask)

    return gradients_penalty


@LOSSES.register_module()
class GradientPenaltyLoss(nn.Module):
    """Gradient penalty loss for wgan-gp.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, discriminator, real_data, fake_data, mask=None):
        """Forward function.

        Args:
            discriminator (nn.Module): Network for the discriminator.
            real_data (Tensor): Real input data.
            fake_data (Tensor): Fake input data.
            mask (Tensor): Masks for inpainting. Default: None.

        Returns:
            Tensor: Loss.
        """
        loss = gradient_penalty_loss(
            discriminator, real_data, fake_data, mask=mask)

        return loss * self.loss_weight


@LOSSES.register_module()
class DiscShiftLoss(nn.Module):
    """Disc shift loss.

    Args:
        loss_weight (float, optional): Loss weight. Defaults to 1.0.
    """

    def __init__(self, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (n, c, h, w)

        Returns:
            Tensor: Loss.
        """
        loss = torch.mean(x**2)

        return loss * self.loss_weight
