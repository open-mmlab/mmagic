# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmengine import MessageHub
from mmengine.optim import OptimWrapperDict

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList
from ...base_models import BaseTranslationModel
from ...utils import set_requires_grad
from .cyclegan_modules import GANImageBuffer


@MODELS.register_module()
class CycleGAN(BaseTranslationModel):
    """CycleGAN model for unpaired image-to-image translation.

    Ref:
    Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
    Networks
    """

    def __init__(self,
                 *args,
                 buffer_size=50,
                 loss_config=dict(cycle_loss_weight=10., id_loss_weight=0.5),
                 **kwargs):
        super().__init__(*args, **kwargs)
        # GAN image buffers
        self.image_buffers = dict()
        self.buffer_size = buffer_size
        for domain in self._reachable_domains:
            self.image_buffers[domain] = GANImageBuffer(self.buffer_size)

        self.loss_config = loss_config

    def forward_test(self, img, target_domain, **kwargs):
        """Forward function for testing.

        Args:
            img (tensor): Input image tensor.
            target_domain (str): Target domain of output image.
            kwargs (dict): Other arguments.

        Returns:
            dict: Forward results.
        """
        # This is a trick for CycleGAN
        # ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e1bdf46198662b0f4d0b318e24568205ec4d7aee/test.py#L54 # noqa
        self.train()
        target = self.translation(img, target_domain=target_domain, **kwargs)
        results = dict(source=img, target=target)
        return results

    def _get_disc_loss(self, outputs):
        """Backward function for the discriminators.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Discriminators' loss and loss dict.
        """
        discriminators = self.get_module(self.discriminators)

        log_vars_d = dict()
        loss_d = 0

        # GAN loss for discriminators['a']
        for domain in self._reachable_domains:
            losses = dict()
            fake_img = self.image_buffers[domain].query(
                outputs[f'fake_{domain}'])
            fake_pred = discriminators[domain](fake_img.detach())
            losses[f'loss_gan_d_{domain}_fake'] = F.mse_loss(
                fake_pred, 0. * torch.ones_like(fake_pred))
            real_pred = discriminators[domain](outputs[f'real_{domain}'])
            losses[f'loss_gan_d_{domain}_real'] = F.mse_loss(
                real_pred, 1. * torch.ones_like(real_pred))

            _loss_d, _log_vars_d = self.parse_losses(losses)
            _loss_d *= 0.5
            loss_d += _loss_d
            log_vars_d[f'loss_gan_d_{domain}'] = _log_vars_d['loss'] * 0.5

        return loss_d, log_vars_d

    def _get_gen_loss(self, outputs):
        """Backward function for the generators.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Generators' loss and loss dict.
        """
        generators = self.get_module(self.generators)
        discriminators = self.get_module(self.discriminators)

        losses = dict()
        # gan loss
        for domain in self._reachable_domains:
            # Identity reconstruction for generators
            outputs[f'identity_{domain}'] = generators[domain](
                outputs[f'real_{domain}'])
            # GAN loss for generators
            fake_pred = discriminators[domain](outputs[f'fake_{domain}'])
            # LSGAN loss
            losses[f'loss_gan_g_{domain}'] = F.mse_loss(
                fake_pred, 1. * torch.ones_like(fake_pred))

        # cycle loss
        loss_weight = self.loss_config['cycle_loss_weight']
        losses['cycle_loss'] = 0.
        for domain in self._reachable_domains:
            losses['cycle_loss'] += loss_weight * F.l1_loss(
                outputs[f'cycle_{domain}'],
                outputs[f'real_{domain}'],
                reduction='mean')

        # id loss
        loss_weight = self.loss_config['id_loss_weight']
        if loss_weight != 0.:
            losses['id_loss'] = 0.
            for domain in self._reachable_domains:
                losses['id_loss'] += loss_weight * F.l1_loss(
                    outputs[f'identity_{domain}'],
                    outputs[f'real_{domain}'],
                    reduction='mean')

        loss_g, log_vars_g = self.parse_losses(losses)

        return loss_g, log_vars_g

    def _get_opposite_domain(self, domain):
        """Get the opposite domain respect to the input domain.

        Args:
            domain (str): The input domain.

        Returns:
            str: The opposite domain.
        """
        for item in self._reachable_domains:
            if item != domain:
                return item
        return None

    def train_step(self, data: dict, optim_wrapper: OptimWrapperDict):
        """Training step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generators and discriminators.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        data = self.data_preprocessor(data, True)
        disc_optimizer_wrapper = optim_wrapper['discriminators']

        inputs_dict = data['inputs']
        outputs, log_vars = dict(), dict()

        # forward generators
        with disc_optimizer_wrapper.optim_context(self.discriminators):
            for target_domain in self._reachable_domains:
                # fetch data by domain
                source_domain = self.get_other_domains(target_domain)[0]
                img = inputs_dict[f'img_{source_domain}']
                # translation process
                results = self(
                    img, test_mode=False, target_domain=target_domain)
                outputs[f'real_{source_domain}'] = results['source']
                outputs[f'fake_{target_domain}'] = results['target']
                # cycle process
                results = self(
                    results['target'],
                    test_mode=False,
                    target_domain=source_domain)
                outputs[f'cycle_{source_domain}'] = results['target']

            # update discriminators
            disc_accu_iters = disc_optimizer_wrapper._accumulative_counts
            loss_d, log_vars_d = self._get_disc_loss(outputs)
            disc_optimizer_wrapper.update_params(loss_d)
            log_vars.update(log_vars_d)

        # generators, no updates to discriminator parameters.
        if ((curr_iter + 1) % (self.discriminator_steps * disc_accu_iters) == 0
                and curr_iter >= self.disc_init_steps):
            set_requires_grad(self.discriminators, False)
            # update generator
            gen_optimizer_wrapper = optim_wrapper['generators']
            with gen_optimizer_wrapper.optim_context(self.generators):
                loss_g, log_vars_g = self._get_gen_loss(outputs)
                gen_optimizer_wrapper.update_params(loss_g)
                log_vars.update(log_vars_g)

            set_requires_grad(self.discriminators, True)

        return log_vars

    def test_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More details in `Metrics` and `Evaluator`.

        Returns:
            SampleList: A list of ``DataSample`` contain generated results.
        """
        data = self.data_preprocessor(data)
        inputs_dict, data_samples = data['inputs'], data['data_samples']

        outputs = {}
        for src_domain in self._reachable_domains:
            # Identity reconstruction for generators
            target_domain = self.get_other_domains(src_domain)[0]
            target = self.forward_test(
                inputs_dict[f'img_{src_domain}'],
                target_domain=target_domain)['target']
            outputs[f'img_{target_domain}'] = target

        batch_sample_list = []
        num_batches = next(iter(outputs.values())).shape[0]
        data_samples = data_samples.split()
        for idx in range(num_batches):
            gen_sample = DataSample()
            if data_samples:
                gen_sample.update(data_samples[idx])

            for src_domain in self._reachable_domains:
                target_domain = self.get_other_domains(src_domain)[0]
                fake_img = outputs[f'img_{target_domain}'][idx]
                fake_img = self.data_preprocessor.destruct(
                    fake_img, data_samples[idx], f'img_{target_domain}')
                setattr(gen_sample, f'fake_{target_domain}', fake_img)

            batch_sample_list.append(gen_sample)
        return batch_sample_list

    def val_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More details in `Metrics` and `Evaluator`.

        Returns:
            SampleList: A list of ``DataSample`` contain generated results.
        """
        data = self.data_preprocessor(data)
        inputs_dict, data_samples = data['inputs'], data['data_samples']

        outputs = {}
        for src_domain in self._reachable_domains:
            # Identity reconstruction for generators
            target_domain = self.get_other_domains(src_domain)[0]
            target = self.forward_test(
                inputs_dict[f'img_{src_domain}'],
                target_domain=target_domain)['target']
            outputs[f'img_{target_domain}'] = target

        batch_sample_list = []
        num_batches = next(iter(outputs.values())).shape[0]
        data_samples = data_samples.split()
        for idx in range(num_batches):
            gen_sample = DataSample()
            if data_samples:
                gen_sample.update(data_samples[idx])

            for src_domain in self._reachable_domains:
                target_domain = self.get_other_domains(src_domain)[0]
                fake_img = outputs[f'img_{target_domain}'][idx]
                fake_img = self.data_preprocessor.destruct(
                    fake_img, data_samples[idx], f'img_{target_domain}')
                gen_sample.set_tensor_data({f'fake_{target_domain}': fake_img})

            batch_sample_list.append(gen_sample)
        return batch_sample_list
