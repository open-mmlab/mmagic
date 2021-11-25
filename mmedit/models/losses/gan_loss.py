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
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

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

    def forward(self, input, target_is_real, is_disc=False, **kwargs):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
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
            mask = kwargs['mask']

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
                    target_label = gaussian_blur(
                        mask, kernel_size=(71, 71),
                        sigma=(10, 10)).detach().cuda()
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
    r"""Creates an operator that blurs a tensor using a Gaussian filter.
    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.
    Arguments:
      kernel_size (Tuple[int, int]): the size of the kernel.
      sigma (Tuple[float, float]): the standard deviation of the kernel.
    Returns:
      Tensor: the blurred tensor.
    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`

    Examples::
      >>> input = torch.rand(2, 4, 5, 5)
      >>> gauss = kornia.filters.GaussianBlur((3, 3), (1.5, 1.5))
      >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._padding = self.compute_zero_padding(kernel_size)
        self.kernel = get_gaussian_kernel2d(kernel_size, sigma)

    @staticmethod
    def compute_zero_padding(kernel_size):
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x):  # type: ignore
        if not torch.is_tensor(x):
            raise TypeError(
                'Input x type is not a torch.Tensor. Got {}'.format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError(
                'Invalid input shape, we expect BxCxHxW. Got: {}'.format(
                    x.shape))
        # prepare kernel
        b, c, h, w = x.shape
        tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # TODO: explore solution when using jit.trace since it raises a warning
        # because the shape is converted to a tensor instead to a int.
        # convolve tensor with gaussian kernel
        return conv2d(x, kernel, padding=self._padding, stride=1, groups=c)


def gaussian(window_size, sigma):

    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)

    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.
    Args:
      kernel_size (int): filter size. It should be odd and positive.
      sigma (float): gaussian standard deviation.
    Returns:
      Tensor: 1D tensor with gaussian filter coefficients.
    Shape:
      - Output: :math:`(\text{kernel_size})`

    Examples::
      >>> kornia.image.get_gaussian_kernel(3, 2.5)
      tensor([0.3243, 0.3513, 0.3243])
      >>> kornia.image.get_gaussian_kernel(5, 1.5)
      tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size,
                      int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError(
            'kernel_size must be an odd positive integer. Got {}'.format(
                kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(kernel_size, sigma):
    r"""Function that returns Gaussian filter matrix coefficients.
    Args:
      kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
        Sizes should be odd and positive.
      sigma (Tuple[int, int]): gaussian standard deviation in the x and y
        direction.
    Returns:
      Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
      - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::
      >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
      tensor([[0.0947, 0.1183, 0.0947],
              [0.1183, 0.1478, 0.1183],
              [0.0947, 0.1183, 0.0947]])

      >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
      tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
              [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
              [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            'kernel_size must be a tuple of length two. Got {}'.format(
                kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            'sigma must be a tuple of length two. Got {}'.format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1),
        kernel_y.unsqueeze(-1).t())
    return kernel_2d


def gaussian_blur(input, kernel_size, sigma):
    r"""Function that blurs a tensor using a Gaussian filter.
    See :class:`~kornia.filters.GaussianBlur` for details.
    """
    return GaussianBlur(kernel_size, sigma)(input)


@LOSSES.register_module()
class SMGANLoss():

    def __init__(self, loss_weight, ksize=71):
        self.ksize = ksize
        self.loss_fn = nn.MSELoss()
        self.loss_weight = loss_weight

    def __call__(self, netD, fake, real, masks, is_real=False):
        fake_detach = fake.detach()

        g_fake = netD(fake)
        d_fake = netD(fake_detach)
        d_real = netD(real)

        _, _, h, w = g_fake.size()
        b, c, ht, wt = masks.size()

        # Handle inconsistent size between outputs and masks
        if h != ht or w != wt:
            g_fake = F.interpolate(
                g_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_fake = F.interpolate(
                d_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_real = F.interpolate(
                d_real, size=(ht, wt), mode='bilinear', align_corners=True)
        d_fake_label = gaussian_blur(masks, (self.ksize, self.ksize),
                                     (10, 10)).detach().cuda()
        d_real_label = torch.zeros_like(d_real).cuda()
        g_fake_label = torch.ones_like(g_fake).cuda()

        dis_loss = self.loss_fn(d_fake, d_fake_label) + self.loss_fn(
            d_real, d_real_label)
        gen_loss = self.loss_fn(g_fake,
                                g_fake_label) * masks / torch.mean(masks)

        if is_real:
            return self.loss_weight * gen_loss.mean()
        return self.loss_weight * dis_loss.mean()


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
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

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
