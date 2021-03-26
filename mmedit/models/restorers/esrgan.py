import torch

from ..common import set_requires_grad
from ..registry import MODELS
from .srgan import SRGAN


@MODELS.register_module()
class ESRGAN(SRGAN):
    """Enhanced SRGAN model for single image super-resolution.

    Ref:
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    It uses RaGAN for GAN updates:
    The relativistic discriminator: a key element missing from standard GAN.

    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator. Default: None.
        gan_loss (dict): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict): Config for the pixel loss. Default: None.
        perceptual_loss (dict): Config for the perceptual loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generate
            update;
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']

        # generator
        fake_g_output = self.generator(lq)

        losses = dict()
        log_vars = dict()

        # no updates to discriminator parameters.
        set_requires_grad(self.discriminator, False)

        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_g_output, gt)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style

            # gan loss for generator
            real_d_pred = self.discriminator(gt).detach()
            fake_g_pred = self.discriminator(fake_g_output)
            loss_gan_fake = self.gan_loss(
                fake_g_pred - torch.mean(real_d_pred),
                target_is_real=True,
                is_disc=False)
            loss_gan_real = self.gan_loss(
                real_d_pred - torch.mean(fake_g_pred),
                target_is_real=False,
                is_disc=False)
            losses['loss_gan'] = (loss_gan_fake + loss_gan_real) / 2

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        # discriminator
        set_requires_grad(self.discriminator, True)
        # real
        fake_d_pred = self.discriminator(fake_g_output).detach()
        real_d_pred = self.discriminator(gt)
        loss_d_real = self.gan_loss(
            real_d_pred - torch.mean(fake_d_pred),
            target_is_real=True,
            is_disc=True
        ) * 0.5  # 0.5 for averaging loss_d_real and loss_d_fake
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_real=loss_d_real))
        optimizer['discriminator'].zero_grad()
        loss_d.backward()
        log_vars.update(log_vars_d)
        # fake
        fake_d_pred = self.discriminator(fake_g_output.detach())
        loss_d_fake = self.gan_loss(
            fake_d_pred - torch.mean(real_d_pred.detach()),
            target_is_real=False,
            is_disc=True
        ) * 0.5  # 0.5 for averaging loss_d_real and loss_d_fake
        loss_d, log_vars_d = self.parse_losses(dict(loss_d_fake=loss_d_fake))
        loss_d.backward()
        log_vars.update(log_vars_d)

        optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=fake_g_output.cpu()))

        return outputs
