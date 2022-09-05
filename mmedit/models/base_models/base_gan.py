# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine import Config, MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from mmedit.registry import MODELS, MODULES
from mmedit.structures import EditDataSample, PixelData
from mmedit.utils.typing import ForwardInputs, LabelVar, NoiseVar, SampleList
from ..utils import (gather_log_vars, get_valid_noise_size,
                     get_valid_num_batches, label_sample_fn, noise_sample_fn,
                     set_requires_grad)

ModelType = Union[Dict, nn.Module]


class BaseGAN(BaseModel, metaclass=ABCMeta):
    """Base class for GAN models.

    Args:
        generator (ModelType): The config or model of the generator.
        discriminator (Optional[ModelType]): The config or model of the
            discriminator. Defaults to None.
        data_preprocessor (Optional[Union[dict, Config]]): The pre-process
            config or :class:`~mmgen.models.GANDataPreprocessor`.
        generator_steps (int): The number of times the generator is completely
            updated before the discriminator is updated. Defaults to 1.
        discriminator_steps (int): The number of times the discriminator is
            completely updated before the generator is updated. Defaults to 1.
        ema_config (Optional[Dict]): The config for generator's exponential
            moving average setting. Defaults to None.
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 noise_size: Optional[int] = None,
                 ema_config: Optional[Dict] = None,
                 loss_config: Optional[Dict] = None):
        super().__init__(data_preprocessor=data_preprocessor)

        # get valid noise_size
        self.noise_size = get_valid_noise_size(noise_size, generator)

        # build generator
        if isinstance(generator, dict):
            self._gen_cfg = deepcopy(generator)
            # build generator with default `noise_size` and `num_classes`
            gen_args = dict()
            if self.noise_size:
                gen_args['noise_size'] = self.noise_size
            if hasattr(self, 'num_classes'):
                gen_args['num_classes'] = self.num_classes
            generator = MODULES.build(generator, default_args=gen_args)
        self.generator = generator

        # build discriminator
        if discriminator:
            if isinstance(discriminator, dict):
                self._disc_cfg = deepcopy(discriminator)
                # build discriminator with default `num_classes`
                disc_args = dict()
                if hasattr(self, 'num_classes'):
                    disc_args['num_classes'] = self.num_classes
                discriminator = MODULES.build(
                    discriminator, default_args=disc_args)
        self.discriminator = discriminator

        self._gen_steps = generator_steps
        self._disc_steps = discriminator_steps

        if ema_config is None:
            self._ema_config = None
            self._with_ema_gen = False
        else:
            self._ema_config = deepcopy(ema_config)
            self._init_ema_model(self._ema_config)
            self._with_ema_gen = True

        self._init_loss(loss_config)

    def _init_loss(self, loss_config: Optional[Dict] = None) -> None:
        """Initialize customized loss modules.

        If loss_config is a dict, we allow kinds of value for each field.

        1. `loss_config` is None: Users will implement all loss calculations
            in their own function. Weights for each loss terms are hard coded.
        2. `loss_config` is dict of scalar or string: Users will implement all
            loss calculations and use passed `loss_config` to control the
            weight or behavior of the loss calculation. Users will unpack and
            use each field in this dict by themself.

            loss_config = dict(gp_norm_mode='HWC', gp_loss_weight=10)

        3. `loss_config` is dict of dict: Each field in `loss_config` will
            used to build a corresponding loss module. And use loss calculation
            function predefined by :class:`BaseGAN` to calculate the loss.

            loss_config = dict()

        Example:
            loss_config = dict(
                # `BaseGAN` pre-defined fields
                gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
                disc_auxiliary_loss=dict(
                    type='R1GradientPenalty',
                    loss_weight=10. / 2.,
                    interval=2,
                    norm_mode='HWC',
                    data_info=dict(
                        real_data='real_imgs',
                        discriminator='disc')),
                gen_auxiliary_loss=dict(
                    type='GeneratorPathRegularizer',
                    loss_weight=2,
                    pl_batch_shrink=2,
                    interval=g_reg_interval,
                    data_info=dict(
                        generator='gen',
                        num_batches='batch_size')),
                # user-defined field for loss weights or loss calculation
                my_loss_2=dict(weight=2, norm_mode='L1'),
                my_loss_3=2,
                my_loss_4_norm_type='L2')


        Args:
            loss_config (Optional[Dict], optional): Loss config used to build
                loss modules or define the loss weights. Defaults to None.
        """
        if loss_config is None:
            self.gan_loss = None
            self.gen_auxiliary_losses = None
            self.disc_auxiliary_losses = None
            self.loss_config = None
            return

        self.loss_config = deepcopy(loss_config)

        # build pre-defined losses
        gan_loss = loss_config.get('gan_loss', None)
        if gan_loss is not None:
            self.gan_loss = MODULES.build(loss_config)
        else:
            self.gan_loss = None

        disc_auxiliary_loss = loss_config.get('disc_auxiliary_loss', None)
        if 'disc_auxiliary_loss' in loss_config:
            self.disc_auxiliary_losses = MODULES.build(disc_auxiliary_loss)
            if not isinstance(self.disc_auxiliary_losses, nn.ModuleList):
                self.disc_auxiliary_losses = nn.ModuleList(
                    [self.disc_auxiliary_losses])
        else:
            self.disc_auxiliary_losses = None

        gen_auxiliary_loss = loss_config.get('gen_auxiliary_loss', None)
        if gen_auxiliary_loss:
            self.gen_auxiliary_losses = MODULES.build(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                self.gen_auxiliary_losses = nn.ModuleList(
                    [self.gen_auxiliary_losses])
        else:
            self.gen_auxiliary_losses = None

    def noise_fn(self, noise: NoiseVar = None, num_batches: int = 1):
        """Sampling function for noise. There are three scenarios in this
        function:

        - If `noise` is a callable function, sample `num_batches` of noise
          with passed `noise`.
        - If `noise` is `None`, sample `num_batches` of noise from gaussian
          distribution.
        - If `noise` is a `torch.Tensor`, directly return `noise`.

        Args:
            noise (Union[Tensor, Callable, List[int], None]): You can directly
                give a batch of label through a ``torch.Tensor`` or offer a
                callable function to sample a batch of label data. Otherwise,
                the ``None`` indicates to use the default noise sampler.
                Defaults to `None`.
            num_batches (int, optional): The number of batches label want to
                sample. If `label` is a Tensor, this will be ignored. Defaults
                to 1.

        Returns:
            Tensor: Sampled noise tensor.
        """
        return noise_sample_fn(
            noise=noise,
            num_batches=num_batches,
            noise_size=self.noise_size,
            device=self.device)

    @property
    def generator_steps(self) -> int:
        """int: The number of times the generator is completely updated before
        the discriminator is updated."""
        return self._gen_steps

    @property
    def discriminator_steps(self) -> int:
        """int: The number of times the discriminator is completely updated
        before the generator is updated."""
        return self._disc_steps

    @property
    def device(self) -> torch.device:
        """Get current device of the model.

        Returns:
            torch.device: The current device of the model.
        """
        return next(self.parameters()).device

    @property
    def with_ema_gen(self) -> bool:
        """Whether the GAN adopts exponential moving average.

        Returns:
            bool: If `True`, means this GAN model is adopted to exponential
                moving average and vice versa.
        """
        return self._with_ema_gen

    def _init_ema_model(self, ema_config: dict):
        """Initialize a EMA model corresponding to the given `ema_config`. If
        `ema_config` is an empty dict or `None`, EMA model will not be
        initialized.

        Args:
            ema_config (dict): Config to initialize the EMA model.
        """
        ema_config.setdefault('type', 'ExponentialMovingAverage')
        self.ema_start = ema_config.pop('start_iter', 0)
        src_model = self.generator.module if is_model_wrapper(
            self.generator) else self.generator
        self.generator_ema = MODELS.build(
            ema_config, default_args=dict(model=src_model))

    def _get_valid_model(self, batch_inputs: ForwardInputs) -> str:
        """Try to get the valid forward model from inputs.

        - If forward model is defined in `batch_inputs`, it will be used as
          forward model.
        - If forward model is not defined in `batch_inputs`, 'ema' will
          returned if :property:`with_ema_gen` is true. Otherwise, 'orig' will
          be returned.

        Args:
            batch_inputs (ForwardInputs): Inputs passed to :meth:`forward`.

        Returns:
            str: Forward model to generate image. ('orig', 'ema' or
                'ema/orig').
        """
        if isinstance(batch_inputs, dict):
            sample_model = batch_inputs.get('sample_model', None)
        else:  # batch_inputs is a Tensor
            sample_model = None

        # set default value
        if sample_model is None:
            if self.with_ema_gen:
                sample_model = 'ema'
            else:
                sample_model = 'orig'

        # security checking for mode
        assert sample_model in [
            'ema', 'ema/orig', 'orig'
        ], ('Only support \'ema\', \'ema/orig\', \'orig\' '
            f'in {self.__class__.__name__}\'s image sampling.')
        if sample_model in ['ema', 'ema/orig']:
            assert self.with_ema_gen, (
                f'\'{self.__class__.__name__}\' do not have EMA model.')
        return sample_model

    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> SampleList:
        """Sample images with the given inputs. If forward mode is 'ema' or
        'orig', the image generated by corresponding generator will be
        returned. If forward mode is 'ema/orig', images generated by original
        generator and EMA generator will both be returned in a dict.

        Args:
            batch_inputs (ForwardInputs): Dict containing the necessary
                information (e.g. noise, num_batches, mode) to generate image.
            data_samples (Optional[list]): Data samples collated by
                :attr:`data_preprocessor`. Defaults to None.
            mode (Optional[str]): `mode` is not used in :class:`BaseGAN`.
                Defaults to None.

        Returns:
            SampleList: A list of ``EditDataSample`` contain generated results.
        """
        if isinstance(inputs, Tensor):
            noise = inputs
            sample_kwargs = {}
        else:
            noise = inputs.get('noise', None)
            num_batches = get_valid_num_batches(inputs)
            noise = self.noise_fn(noise, num_batches=num_batches)
            sample_kwargs = inputs.get('sample_kwargs', dict())
        num_batches = noise.shape[0]

        sample_model = self._get_valid_model(inputs)
        if sample_model in ['ema', 'ema/orig']:
            generator = self.generator_ema
        else:  # sample model is 'orig'
            generator = self.generator

        num_batches = noise.shape[0]
        outputs = generator(noise, return_noise=False, **sample_kwargs)

        if sample_model == 'ema/orig':
            generator = self.generator
            outputs_orig = generator(
                noise, return_noise=False, **sample_kwargs)
            outputs = dict(ema=outputs, orig=outputs_orig)

        batch_sample_list = []
        for idx in range(num_batches):
            gen_sample = EditDataSample()
            if data_samples:
                gen_sample.update(data_samples[idx])
            if isinstance(inputs, dict) and 'img' in inputs:
                gen_sample.gt_img = PixelData(data=inputs['img'][idx])
            if isinstance(outputs, dict):
                gen_sample.ema = EditDataSample(
                    fake_img=PixelData(data=outputs['ema'][idx]),
                    sample_model='ema')
                gen_sample.orig = EditDataSample(
                    fake_img=PixelData(data=outputs['orig'][idx]),
                    sample_model='orig')
                gen_sample.sample_model = 'ema/orig'
            else:
                gen_sample.fake_img = PixelData(data=outputs[idx])
                gen_sample.sample_model = sample_model

            # Append input condition (noise and sample_kwargs) to
            # batch_sample_list
            gen_sample.noise = noise[idx]
            gen_sample.sample_kwargs = deepcopy(sample_kwargs)

            batch_sample_list.append(gen_sample)

        return batch_sample_list

    def val_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data.

        Calls ``self.data_preprocessor(data)`` and
        ``self(inputs, data_sample, mode=None)`` in order. Return the
        generated results which will be passed to evaluator.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            SampleList: Generated image or image dict.
        """
        data = self.data_preprocessor(data)
        outputs = self(**data)
        return outputs

    def test_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            List[EditDataSample]: Generated image or image dict.
        """
        data = self.data_preprocessor(data)
        outputs = self(**data)
        return outputs

    def train_step(self, data: dict,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        """Train GAN model. In the training of GAN models, generator and
        discriminator are updated alternatively. In MMGeneration's design,
        `self.train_step` is called with data input. Therefore we always update
        discriminator, whose updating is relay on real data, and then determine
        if the generator needs to be updated based on the current number of
        iterations. More details about whether to update generator can be found
        in :meth:`should_gen_update`.

        Args:
            data (dict): Data sampled from dataloader.
            optim_wrapper (OptimWrapperDict): OptimWrapperDict instance
                contains OptimWrapper of generator and discriminator.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        data = self.data_preprocessor(data, True)

        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(
                **data, optimizer_wrapper=disc_optimizer_wrapper)

        # add 1 to `curr_iter` because iter is updated in train loop.
        # Whether to update the generator. We update generator with
        # discriminator is fully updated for `self.n_discriminator_steps`
        # iterations. And one full updating for discriminator contains
        # `disc_accu_counts` times of grad accumulations.
        if (curr_iter + 1) % (self.discriminator_steps * disc_accu_iters) == 0:
            set_requires_grad(self.discriminator, False)
            gen_optimizer_wrapper = optim_wrapper['generator']
            gen_accu_iters = gen_optimizer_wrapper._accumulative_counts

            log_vars_gen_list = []
            # init optimizer wrapper status for generator manually
            gen_optimizer_wrapper.initialize_count_status(
                self.generator, 0, self.generator_steps * gen_accu_iters)
            for _ in range(self.generator_steps * gen_accu_iters):
                with gen_optimizer_wrapper.optim_context(self.generator):
                    log_vars_gen = self.train_generator(
                        **data, optimizer_wrapper=gen_optimizer_wrapper)

                log_vars_gen_list.append(log_vars_gen)
            log_vars_gen = gather_log_vars(log_vars_gen_list)
            log_vars_gen.pop('loss', None)  # remove 'loss' from gen logs

            set_requires_grad(self.discriminator, True)

            # only do ema after generator update
            if self.with_ema_gen and (curr_iter + 1) >= (
                    self.ema_start * self.discriminator_steps *
                    disc_accu_iters):
                self.generator_ema.update_parameters(
                    self.generator.module
                    if is_model_wrapper(self.generator) else self.generator)
                # if not update buffer, copy buffer from orig model
                if not self.generator_ema.update_buffers:
                    self.generator_ema.sync_buffers(
                        self.generator.module if is_model_wrapper(
                            self.generator) else self.generator)
            elif self.with_ema_gen:
                # before ema, copy weights from orig
                self.generator_ema.sync_parameters(
                    self.generator.module
                    if is_model_wrapper(self.generator) else self.generator)

            log_vars.update(log_vars_gen)

        return log_vars

    def _get_gen_loss(self, out_dict):
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_fake_g'] = self.gan_loss(
            out_dict['disc_pred_fake_g'], target_is_real=True, is_disc=False)

        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(out_dict)
                if loss_ is None:
                    continue

                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var

    def _get_disc_loss(self, out_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_fake'] = self.gan_loss(
            out_dict['disc_pred_fake'], target_is_real=False, is_disc=True)
        losses_dict['loss_disc_real'] = self.gan_loss(
            out_dict['disc_pred_real'], target_is_real=True, is_disc=True)

        # disc auxiliary loss
        if self.with_disc_auxiliary_loss:
            for loss_module in self.disc_auxiliary_losses:
                loss_ = loss_module(out_dict)
                if loss_ is None:
                    continue

                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self.parse_losses(losses_dict)

        return loss, log_var

    # @abstractmethod
    def train_generator(self, inputs: dict, data_samples: List[EditDataSample],
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[EditDataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        num_batches = inputs['img'].shape[0]
        noise = self.noise_fn(num_batches=num_batches)

        fake_imgs = self.generator(noise=noise)
        disc_pred_fake_g = self.discriminator(fake_imgs)

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake_g,
            # iteration=curr_iter,
            batch_size=num_batches,
            loss_scaler=getattr(optimizer_wrapper, 'loss_scaler', None))
        loss, log_vars = self._get_gen_loss(data_dict_)

        optimizer_wrapper.update_params(loss)
        return log_vars

    def train_discriminator(self, inputs: dict,
                            data_samples: List[EditDataSample],
                            optimizer_wrapper: OptimWrapper
                            ) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[EditDataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        real_imgs, num_batches = inputs['img'], inputs['img'].shape[0]
        noise = self.noise_fn(num_batches=num_batches)
        fake_imgs = self.generator(noise=noise)

        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            # iteration=curr_iter,
            batch_size=num_batches,
            loss_scaler=getattr(optimizer_wrapper, 'loss_scaler', None))
        loss, log_vars = self._get_disc_loss(data_dict_)

        optimizer_wrapper.update_params(loss)
        return log_vars


class BaseConditionalGAN(BaseGAN):
    """Base class for Conditional GAM models.

    Args:
        generator (ModelType): The config or model of the generator.
        discriminator (Optional[ModelType]): The config or model of the
            discriminator. Defaults to None.
        data_preprocessor (Optional[Union[dict, Config]]): The pre-process
            config or :class:`~mmgen.models.GANDataPreprocessor`.
        generator_steps (int): The number of times the generator is completely
            updated before the discriminator is updated. Defaults to 1.
        discriminator_steps (int): The number of times the discriminator is
            completely updated before the generator is updated. Defaults to 1.
        noise_size (Optional[int]): Size of the input noise vector.
            Default to None.
        num_classes (Optional[int]): The number classes you would like to
            generate. Defaults to None.
        ema_config (Optional[Dict]): The config for generator's exponential
            moving average setting. Defaults to None.
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 noise_size: Optional[int] = None,
                 num_classes: Optional[int] = None,
                 ema_config: Optional[Dict] = None):

        self.num_classes = self._get_valid_num_classes(num_classes, generator,
                                                       discriminator)
        super().__init__(generator, discriminator, data_preprocessor,
                         generator_steps, discriminator_steps, noise_size,
                         ema_config)

    def label_fn(self, label: LabelVar = None, num_batches: int = 1) -> Tensor:
        """Sampling function for label. There are three scenarios in this
        function:

        - If `label` is a callable function, sample `num_batches` of labels
          with passed `label`.
        - If `label` is `None`, sample `num_batches` of labels in range of
          `[0, self.num_classes-1]` uniformly.
        - If `label` is a `torch.Tensor`, check the range of the tensor is in
          `[0, self.num_classes-1]`. If all values are in valid range,
          directly return `label`.

        Args:
            label (Union[Tensor, Callable, List[int], None]): You can directly
                give a batch of label through a ``torch.Tensor`` or offer a
                callable function to sample a batch of label data. Otherwise,
                the ``None`` indicates to use the default label sampler.
                Defaults to `None`.
            num_batches (int, optional): The number of batches label want to
                sample. If `label` is a Tensor, this will be ignored. Defaults
                to 1.

        Returns:
            Tensor: Sampled label tensor.
        """
        return label_sample_fn(
            label=label,
            num_batches=num_batches,
            num_classes=self.num_classes,
            device=self.device)

    def data_sample_to_label(self,
                             data_sample: list) -> Optional[torch.Tensor]:
        """Get labels from input `data_sample` and pack to `torch.Tensor`. If
        no label is found in the passed `data_sample`, `None` would be
        returned.

        Args:
            data_sample (List[InstanceData]): Input data samples.

        Returns:
            Optional[torch.Tensor]: Packed label tensor.
        """
        # assume all data_sample have the same data fields
        if not data_sample or 'gt_label' not in data_sample[0].keys():
            return None
        gt_labels = [sample.gt_label.label for sample in data_sample]
        gt_labels = torch.cat(gt_labels, dim=0)
        return gt_labels

    @staticmethod
    def _get_valid_num_classes(num_classes: Optional[int],
                               generator: ModelType,
                               discriminator: Optional[ModelType]) -> int:
        """Try to get the value of `num_classes` from input, `generator` and
        `discriminator` and check the consistency of these values. If no
        conflict is found, return the `num_classes`.

        Args:
            num_classes (Optional[int]): `num_classes` passed to
                `BaseConditionalGAN_refactor`'s initialize function.
            generator (ModelType): The config or the model of generator.
            discriminator (Optional[ModelType]): The config or model of
                discriminator.

        Returns:
            int: The number of classes to be generated.
        """
        if isinstance(generator, dict):
            num_classes_gen = generator.get('num_classes', None)
        else:
            num_classes_gen = getattr(generator, 'num_classes', None)

        num_classes_disc = None
        if discriminator is not None:
            if isinstance(discriminator, dict):
                num_classes_disc = discriminator.get('num_classes', None)
            else:
                num_classes_disc = getattr(discriminator, 'num_classes', None)

        # check consistency between gen and disc
        if num_classes_gen is not None and num_classes_disc is not None:
            assert num_classes_disc == num_classes_gen, (
                '\'num_classes\' is unconsistency between generator and '
                f'discriminator. Receive \'{num_classes_gen}\' and '
                f'\'{num_classes_disc}\'.')
        model_num_classes = num_classes_gen or num_classes_disc

        if num_classes is not None and model_num_classes is not None:
            assert num_classes == model_num_classes, (
                'Input \'num_classes\' is unconsistency with '
                f'model\'s ones. Receive \'{num_classes}\' and '
                f'\'{model_num_classes}\'.')

        num_classes = num_classes or model_num_classes
        return num_classes

    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> List[EditDataSample]:
        """Sample images with the given inputs. If forward mode is 'ema' or
        'orig', the image generated by corresponding generator will be
        returned. If forward mode is 'ema/orig', images generated by original
        generator and EMA generator will both be returned in a dict.

        Args:
            inputs (ForwardInputs): Dict containing the necessary
                information (e.g. noise, num_batches, mode) to generate image.
            data_samples (Optional[list]): Data samples collated by
                :attr:`data_preprocessor`. Defaults to None.
            mode (Optional[str]): `mode` is not used in
                :class:`BaseConditionalGAN`. Defaults to None.

        Returns:
            List[EditDataSample]: Generated images or image dict.
        """
        if isinstance(inputs, Tensor):
            noise = inputs
            sample_kwargs = {}
        else:
            noise = inputs.get('noise', None)
            num_batches = get_valid_num_batches(inputs)
            noise = self.noise_fn(noise, num_batches=num_batches)
            sample_kwargs = inputs.get('sample_kwargs', dict())
        num_batches = noise.shape[0]

        labels = self.data_sample_to_label(data_samples)
        if labels is None:
            num_batches = get_valid_num_batches(inputs)
            labels = self.label_fn(num_batches=num_batches)

        sample_model = self._get_valid_model(inputs)
        if sample_model in ['ema', 'ema/orig']:
            generator = self.generator_ema
        else:  # sample model is `orig`
            generator = self.generator
        outputs = generator(noise, label=labels, return_noise=False)

        if sample_model == 'ema/orig':
            generator = self.generator
            outputs_orig = generator(noise, label=labels, return_noise=False)

            outputs = dict(ema=outputs, orig=outputs_orig)

        batch_sample_list = []
        for idx in range(num_batches):
            gen_sample = EditDataSample()
            if data_samples:
                gen_sample.update(data_samples[idx])
            if isinstance(outputs, dict):
                gen_sample.ema = EditDataSample(
                    fake_img=PixelData(data=outputs['ema'][idx]),
                    sample_model='ema')
                gen_sample.orig = EditDataSample(
                    fake_img=PixelData(data=outputs['orig'][idx]),
                    sample_model='orig')
                gen_sample.sample_model = 'ema/orig'
                gen_sample.set_gt_label(labels[idx])
                gen_sample.ema.set_gt_label(labels[idx])
                gen_sample.orig.set_gt_label(labels[idx])
            else:
                gen_sample.fake_img = PixelData(data=outputs[idx])
                gen_sample.sample_model = sample_model
                gen_sample.set_gt_label(labels[idx])

            # Append input condition (noise and sample_kwargs) to
            # batch_sample_list
            gen_sample.noise = noise[idx]
            gen_sample.sample_kwargs = deepcopy(sample_kwargs)
            batch_sample_list.append(gen_sample)
        return batch_sample_list

    def train_generator(self, inputs: dict, data_samples: List[EditDataSample],
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[EditDataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        num_batches = inputs.shape[0]

        noise = self.noise_fn(num_batches=num_batches)
        fake_labels = self.label_fn(num_batches=num_batches)
        fake_imgs = self.generator(
            noise=noise, label=fake_labels, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs, label=fake_labels)

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake,
            # iteration=curr_iter,
            batch_size=num_batches,
            fake_label=fake_labels,
            loss_scaler=getattr(optimizer_wrapper, 'loss_scaler', None))
        parsed_loss, log_vars = self._get_gen_loss(data_dict_)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars

    def train_discriminator(self, inputs: dict,
                            data_samples: List[EditDataSample],
                            optimizer_wrapper: OptimWrapper
                            ) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[EditDataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        real_imgs = inputs
        real_labels = self.data_sample_to_label(data_samples)
        assert real_labels is not None, (
            'Cannot found \'gt_label\' in \'data_sample\'.')

        num_batches = real_imgs.shape[0]

        noise_batch = self.noise_fn(num_batches=num_batches)
        fake_labels = self.label_fn(num_batches=num_batches)
        with torch.no_grad():
            fake_imgs = self.generator(
                noise=noise_batch, label=fake_labels, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs, label=fake_labels)
        disc_pred_real = self.discriminator(real_imgs, label=real_labels)

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            # iteration=curr_iter,
            batch_size=num_batches,
            gt_label=real_labels,
            fake_label=fake_labels,
            loss_scaler=setattr(optimizer_wrapper, 'loss_scaler', None))
        loss, log_vars = self._get_disc_loss(data_dict_)

        optimizer_wrapper.update_params(loss)
        return log_vars
