# Copyright (c) OpenMMLab. All rights reserved.
import random
from collections import deque

import torch
import torch.nn as nn

from mmagic.registry import MODELS
from .gan_loss import GANLoss, gradient_penalty_loss


class ImagePool:
    """Defined a image pool for RelativisticDiscLoss."""

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.sample_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = deque()

    def add(self, images):
        if self.pool_size == 0:
            return images
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
            else:
                self.images.popleft()
                self.images.append(image)

    def query(self):
        if len(self.images) > self.sample_size:
            return_images = list(random.sample(self.images, self.sample_size))
        else:
            return_images = list(self.images)
        return torch.cat(return_images, 0)


class DiscLoss(nn.Module):
    """Defined a criterion to calculator loss."""

    def name(self):
        """return name of criterion."""
        return 'DiscLoss'

    def __init__(self, gan_type='vanilla'):
        super(DiscLoss, self).__init__()

        self.criterionGAN = GANLoss(gan_type=gan_type)

    def forward(self, net, fakeB, realB, model='discriminator'):
        """Get discriminator or generator loss."""
        if model == 'discriminator':
            self.pred_fake = net.forward(fakeB.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

            # Real
            self.pred_real = net.forward(realB)
            self.loss_D_real = self.criterionGAN(self.pred_real, 1)

            # Combined loss
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            return self.loss_D
        else:
            pred_fake = net.forward(fakeB)
            return self.criterionGAN(pred_fake, 1)


class RelativisticDiscLoss(nn.Module):
    """Defined a criterion to calculator loss."""

    def name(self):
        """return name of criterion."""
        return 'RelativisticDiscLoss'

    def __init__(self):
        super(RelativisticDiscLoss, self).__init__()

        self.criterionGAN = GANLoss(gan_type='vanilla')
        self.fake_pool = ImagePool(50)
        self.real_pool = ImagePool(50)

    def forward(self, net, fakeB, realB, model='discriminator'):
        """Get discriminator or generator loss."""
        if model == 'discriminator':
            self.fake_B = fakeB.detach()
            self.real_B = realB
            self.pred_fake = net.forward(self.fake_B)
            self.fake_pool.add(self.pred_fake)

            # Real
            self.pred_real = net.forward(self.real_B)
            self.real_pool.add(self.pred_real)

            # Combined loss
            self.loss_D = (self.criterionGAN(
                self.pred_real - torch.mean(self.fake_pool.query()), 1) +
                           self.criterionGAN(
                               self.pred_fake -
                               torch.mean(self.real_pool.query()), 0)) / 2
            return self.loss_D
        else:
            self.pred_fake = net.forward(fakeB)

            # Real
            self.pred_real = net.forward(realB)
            errG = (self.criterionGAN(
                self.pred_real - torch.mean(self.fake_pool.query()), 0) +
                    self.criterionGAN(
                        self.pred_fake - torch.mean(self.real_pool.query()),
                        1)) / 2
            return errG


class RelativisticDiscLossLS(nn.Module):
    """Defined a criterion to calculator loss."""

    def name(self):
        """return name of criterion."""
        return 'RelativisticDiscLossLS'

    def __init__(self):
        super(RelativisticDiscLossLS, self).__init__()

        self.criterionGAN = GANLoss(gan_type='l1')
        self.fake_pool = ImagePool(50)
        self.real_pool = ImagePool(50)

    def forward(self, net, fakeB, realB, model='discriminator'):
        """Get discriminator or generator loss."""
        if model == 'discriminator':
            self.fake_B = fakeB.detach()
            self.real_B = realB
            self.pred_fake = net.forward(fakeB.detach())
            self.fake_pool.add(self.pred_fake)

            # Real
            self.pred_real = net.forward(realB)
            self.real_pool.add(self.pred_real)

            # Combined loss
            ex_pdata = torch.mean(
                (self.pred_real - torch.mean(self.fake_pool.query()) - 1)**2)
            ex_pz = torch.mean(
                (self.pred_fake - torch.mean(self.real_pool.query()) + 1)**2)
            self.loss_D = (ex_pdata + ex_pz) / 2
            return self.loss_D
        else:
            self.pred_fake = net.forward(fakeB)

            # Real
            self.pred_real = net.forward(realB)
            ex_pdata = torch.mean(
                (self.pred_real - torch.mean(self.fake_pool.query()) + 1)**2)
            ez_pz = torch.mean(
                (self.pred_fake - torch.mean(self.real_pool.query()) - 1)**2)
            errG = (ex_pdata + ez_pz) / 2
            return errG


class DiscLossWGANGP(DiscLoss):
    """Defined a criterion to calculator loss."""

    def name(self):
        """return name of criterion."""
        return 'DiscLossWGAN-GP'

    def __init__(self):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = 10

    def forward(self, net, fakeB, realB, model='discriminator'):
        """Get discriminator or generator loss."""
        if model == 'discriminator':
            self.D_fake = net.forward(fakeB.detach())
            self.D_fake = self.D_fake.mean()

            # Real
            self.D_real = net.forward(realB)
            self.D_real = self.D_real.mean()
            # Combined loss
            self.loss_D = self.D_fake - self.D_real
            gradient_penalty = gradient_penalty_loss(net, realB.data,
                                                     fakeB.data) * self.LAMBDA
            return self.loss_D + gradient_penalty
        else:
            self.D_fake = net.forward(fakeB)
            return -self.D_fake.mean()


@MODELS.register_module()
class AdvLoss:
    """Returns the loss of discriminator with the specified type for
    DeblurGanv2.

    Args:
        loss_type (Str): One of value in [wgan-gp,lsgan,gan,ragan,ragan-ls].
    """

    def __new__(cls, loss_type: str):
        if loss_type == 'wgan-gp':
            disc_loss = DiscLossWGANGP()
        elif loss_type == 'lsgan':
            disc_loss = DiscLoss(gan_type='l1')
        elif loss_type == 'gan':
            disc_loss = DiscLoss()
        elif loss_type == 'ragan':
            disc_loss = RelativisticDiscLoss()
        elif loss_type == 'ragan-ls':
            disc_loss = RelativisticDiscLossLS()
        else:
            raise ValueError('GAN Loss [%s] not recognized.' % loss_type)
        return disc_loss
