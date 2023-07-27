# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapperDict
from torch import nn

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from .deblurganv2_util import get_disc_loss, get_pixel_loss

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class DeblurGanV2(BaseModel):

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 pixel_loss: Optional[Union[dict, str]] = None,
                 disc_loss: Optional[Union[dict, str]] = None,
                 adv_lambda: float = 0.001,
                 warmup_num: int = 3,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        # generator
        if isinstance(generator, dict):
            self.generator = MODELS.build(generator)
        else:
            self.generator = generator

        # discriminator
        if discriminator:
            if isinstance(generator, dict):
                self.discriminator = MODELS.build(discriminator)
            else:
                self.discriminator = discriminator
        else:
            self.discriminator = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.epoch_num = 0
        self.warmup_num = warmup_num
        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_repeat = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_repeat', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        self.adv_lambda = adv_lambda

        self.register_buffer('step_counter', torch.tensor(0), False)

        # loss
        # self.gan_loss = None
        # self.perceptual_loss = None
        if isinstance(pixel_loss, dict):
            self.pixel_loss = MODELS.build(pixel_loss)
        elif isinstance(pixel_loss, str):
            self.pixel_loss = get_pixel_loss(pixel_loss)

        if isinstance(disc_loss, dict):
            self.disc_loss = MODELS.build(disc_loss)
        elif isinstance(disc_loss, str):
            self.disc_loss = get_disc_loss(disc_loss)
        else:
            self.disc_loss = None
        if self.disc_loss:
            self.disc_loss2 = copy.deepcopy(self.disc_loss)

        # self.adv_trainer = GANFactory().get_adversarial_trainer(
        #     self.discriminator.d_name, self.discriminator, self.disc_loss)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                **kwargs) -> Union[torch.Tensor, List[DataSample], dict]:
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``inputs`` and ``data_samples`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.val_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``. Default: 'tensor'.

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            ForwardResults:

                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of
                  :obj:`BaseDataElement` for computing metric
                  and getting inference result.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict`` or tensor for custom use.
        """
        if isinstance(inputs, dict):
            inputs = inputs['img']
        if mode == 'tensor':
            return self.forward_tensor(inputs, data_samples, **kwargs)

        elif mode == 'val':
            predictions = self.forward_inference(inputs, data_samples,
                                                 **kwargs)
            predictions = self.convert_to_datasample(predictions, data_samples,
                                                     inputs)
            return predictions
        elif mode == 'predict':
            h, w = data_samples.ori_img_shape[0][0:2]
            block_size = 32
            min_height = (h // block_size + 1) * block_size
            min_width = (w // block_size + 1) * block_size
            pad = torch.nn.ZeroPad2d(
                padding=(0, min_width - w, 0, min_height - h))
            inputs = pad(inputs)
            predictions = self.forward_inference(inputs, data_samples,
                                                 **kwargs)
            predictions.pred_img = predictions.pred_img[:, :, :h, :w]
            predictions = self.convert_to_datasample(predictions, data_samples,
                                                     inputs)
            return predictions

        elif mode == 'loss':
            return self.forward_train(inputs, data_samples, **kwargs)

    def convert_to_datasample(self, predictions: DataSample,
                              data_samples: DataSample,
                              inputs: Optional[torch.Tensor]
                              ) -> List[DataSample]:
        """Add predictions and destructed inputs (if passed) to data samples.

        Args:
            predictions (DataSample): The predictions of the model.
            data_samples (DataSample): The data samples loaded from
                dataloader.
            inputs (Optional[torch.Tensor]): The input of model. Defaults to
                None.

        Returns:
            List[DataSample]: Modified data samples.
        """

        if inputs is not None:
            destructed_input = self.data_preprocessor.destruct(
                inputs, data_samples, 'img')
            data_samples.set_tensor_data({'input': destructed_input})
        # split to list of data samples
        data_samples = data_samples.split()
        predictions = predictions.split()

        for data_sample, pred in zip(data_samples, predictions):
            data_sample.output = pred

        return data_samples

    def forward_tensor(self,
                       inputs: torch.Tensor,
                       data_samples: Optional[List[DataSample]] = None,
                       **kwargs) -> torch.Tensor:
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: result of simple forward.
        """

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # feats = self.generator(img, **kwargs)
        feats = self.generator(inputs)
        # feats = (feats + 1) / 2.0
        return feats

    def forward_inference(self,
                          inputs: torch.Tensor,
                          data_samples: Optional[List[DataSample]] = None,
                          **kwargs) -> List[DataSample]:
        """Forward inference. Returns predictions of validation, testing, and
        simple inference.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            List[EditDataSample]: predictions.
        """

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        feats = self.data_preprocessor.destruct(feats, data_samples)

        # create a stacked data sample here
        predictions = DataSample(pred_img=feats.cpu())

        return predictions

    def forward_train(self, inputs, data_samples=None, **kwargs):
        """Forward training. Losses of training is calculated in train_step.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Tensor: Result of ``forward_tensor`` with ``training=True``.
        """

        return self.forward_tensor(
            inputs, data_samples, training=True, **kwargs)

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='val')  # type: ignore

        self.epoch_num += 1

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapperDict) -> Dict[str, torch.Tensor]:
        """Train step of GAN-based method.

        Args:
            data (List[dict]): Data sampled from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """

        data = self.data_preprocessor(data, True)
        batch_inputs = data['inputs']

        data_samples = data['data_samples']
        batch_gt_data = self.extract_gt_data(data_samples)

        log_vars = dict()

        if self.warmup_num == self.epoch_num:
            self.generator.module.unfreeze()

        g_optim_wrapper = optim_wrapper['generator']
        with g_optim_wrapper.optim_context(self):
            batch_outputs = self.forward_train(batch_inputs, data_samples)

        # if self.if_run_d() or True:
        # set_requires_grad(self.discriminator, True)

        for _ in range(self.disc_repeat):
            # detach before function call to resolve PyTorch2.0 compile bug
            log_vars_d = self.d_step_with_optim(
                batch_outputs=batch_outputs.detach(),
                batch_gt_data=batch_gt_data,
                optim_wrapper=optim_wrapper)

        log_vars.update(log_vars_d)

        # if self.if_run_g() or True:
        # set_requires_grad(self.discriminator, False)

        log_vars_d = self.g_step_with_optim(
            batch_outputs=batch_outputs,
            batch_gt_data=batch_gt_data,
            optim_wrapper=optim_wrapper)

        log_vars.update(log_vars_d)

        if 'loss' in log_vars:
            log_vars.pop('loss')

        self.step_counter += 1

        return log_vars

    def g_step_double(self, batch_outputs: torch.Tensor,
                      batch_gt_data: torch.Tensor):
        """G step of DobuleGAN: Calculate losses of generator.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        losses = dict()

        # pix loss
        if self.pixel_loss:
            losses['loss_g_content'] = self.pixel_loss(batch_outputs,
                                                       batch_gt_data)

        losses['loss_g_adv'] = self.adv_lambda * (self.disc_loss.get_g_loss(
            self.discriminator.patch_gan, batch_outputs,
            batch_gt_data) + self.disc_loss2.get_g_loss(
                self.discriminator.full_gan, batch_outputs, batch_gt_data)) / 2
        losses['loss_g'] = losses['loss_g_content'] + losses['loss_g_adv']

        return losses

    # def d_step_real(self, batch_outputs, batch_gt_data: torch.Tensor):
    #     """Real part of D step.
    #
    #     Args:
    #         batch_outputs (Tensor): Batch output of generator.
    #         batch_gt_data (Tensor): Batch GT data.
    #
    #     Returns:
    #         Tensor: Real part of gan_loss for discriminator.
    #     """
    #
    #     # real
    #     real_d_pred = self.discriminator(batch_gt_data)
    #     loss_d_real = self.gan_loss(
    #         real_d_pred, target_is_real=True, is_disc=True)
    #
    #     return loss_d_real
    #
    # def d_step_fake(self, batch_outputs: torch.Tensor, batch_gt_data):
    #     """Fake part of D step.
    #
    #     Args:
    #         batch_outputs (Tensor): Batch output of generator.
    #         batch_gt_data (Tensor): Batch GT data.
    #
    #     Returns:
    #         Tensor: Fake part of gan_loss for discriminator.
    #     """
    #
    #     # fake
    #     fake_d_pred = self.discriminator(batch_outputs.detach())
    #     loss_d_fake = self.gan_loss(
    #         fake_d_pred, target_is_real=False, is_disc=True)
    #
    #     return loss_d_fake

    def d_step_double(self, batch_outputs: torch.Tensor,
                      batch_gt_data: torch.Tensor):
        """D step of DobuleGAN: Calculate losses of generator.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.

        Returns:
            dict: Dict of losses.
        """
        loss_d_double = (self.disc_loss(self.discriminator.patch_gan,
                                        batch_outputs, batch_gt_data) +
                         self.disc_loss2(self.discriminator.full_gan,
                                         batch_outputs, batch_gt_data)) / 2
        return loss_d_double

    def g_step_with_optim(self, batch_outputs: torch.Tensor,
                          batch_gt_data: torch.Tensor,
                          optim_wrapper: OptimWrapperDict):
        """G step with optim of GAN: Calculate losses of generator and run
        optim.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.
            optim_wrapper (OptimWrapperDict): Optim wrapper dict.

        Returns:
            dict: Dict of parsed losses.
        """

        g_optim_wrapper = optim_wrapper['generator']
        g_optim_wrapper.zero_grad()
        with g_optim_wrapper.optim_context(self):
            losses_g_double = self.g_step_double(batch_outputs, batch_gt_data)

        parsed_losses_g, log_vars_g = self.parse_losses(losses_g_double)
        loss_pix = g_optim_wrapper.scale_loss(parsed_losses_g)
        g_optim_wrapper.backward(loss_pix)
        g_optim_wrapper.step()
        # g_optim_wrapper.update_params(parsed_losses_g)

        return log_vars_g

    def d_step_with_optim(self, batch_outputs: torch.Tensor,
                          batch_gt_data: torch.Tensor,
                          optim_wrapper: OptimWrapperDict):
        """D step with optim of GAN: Calculate losses of discriminator and run
        optim.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.
            optim_wrapper (OptimWrapperDict): Optim wrapper dict.

        Returns:
            dict: Dict of parsed losses.
        """

        log_vars = dict()
        d_optim_wrapper = optim_wrapper['discriminator']

        # with d_optim_wrapper.optim_context(self):
        #     loss_d_real = self.d_step_real(batch_outputs, batch_gt_data)
        #
        # parsed_losses_dr, log_vars_dr = self.parse_losses(
        #     dict(loss_d_real=loss_d_real))
        # log_vars.update(log_vars_dr)
        # loss_dr = d_optim_wrapper.scale_loss(parsed_losses_dr)
        # d_optim_wrapper.backward(loss_dr)

        # with d_optim_wrapper.optim_context(self):
        #     loss_d_fake = self.d_step_fake(batch_outputs, batch_gt_data)
        #
        # parsed_losses_df, log_vars_df = self.parse_losses(
        #     dict(loss_d_fake=loss_d_fake))
        # log_vars.update(log_vars_df)
        # loss_df = d_optim_wrapper.scale_loss(parsed_losses_df)
        # d_optim_wrapper.backward(loss_df)

        d_optim_wrapper.zero_grad()
        with d_optim_wrapper.optim_context(self):

            loss_d_double = self.adv_lambda * self.d_step_double(
                batch_outputs, batch_gt_data)

        parsed_losses_df, log_vars_df = self.parse_losses(
            dict(loss_d=loss_d_double))
        log_vars.update(log_vars_df)
        loss_df = d_optim_wrapper.scale_loss(parsed_losses_df)
        d_optim_wrapper.backward(loss_df, retain_graph=True)
        d_optim_wrapper.step()

        return log_vars

    def extract_gt_data(self, data_samples):
        """extract gt data from data samples.

        Args:
            data_samples (list): List of DataSample.

        Returns:
            Tensor: Extract gt data.
        """

        batch_gt_data = data_samples.gt_img

        return batch_gt_data
