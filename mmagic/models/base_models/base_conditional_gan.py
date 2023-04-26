# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine import Config
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmagic.structures import DataSample
from mmagic.utils.typing import ForwardInputs, LabelVar
from ..utils import get_valid_num_batches, label_sample_fn
from .base_gan import BaseGAN

ModelType = Union[Dict, nn.Module]


class BaseConditionalGAN(BaseGAN):
    """Base class for Conditional GAM models.

    Args:
        generator (ModelType): The config or model of the generator.
        discriminator (Optional[ModelType]): The config or model of the
            discriminator. Defaults to None.
        data_preprocessor (Optional[Union[dict, Config]]): The pre-process
            config or :class:`~mmagic.models.DataPreprocessor`.
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
                 ema_config: Optional[Dict] = None,
                 loss_config: Optional[Dict] = None):

        self.num_classes = self._get_valid_num_classes(num_classes, generator,
                                                       discriminator)
        super().__init__(generator, discriminator, data_preprocessor,
                         generator_steps, discriminator_steps, noise_size,
                         ema_config, loss_config)

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

    def data_sample_to_label(self, data_sample: DataSample
                             ) -> Optional[torch.Tensor]:
        """Get labels from input `data_sample` and pack to `torch.Tensor`. If
        no label is found in the passed `data_sample`, `None` would be
        returned.

        Args:
            data_sample (DataSample): Input data samples.

        Returns:
            Optional[torch.Tensor]: Packed label tensor.
        """
        # assume all data_sample have the same data fields
        if not data_sample or 'gt_label' not in data_sample.keys():
            return None
        gt_labels = data_sample.gt_label.label
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
                '\'num_classes\' is inconsistent between generator and '
                f'discriminator. Receive \'{num_classes_gen}\' and '
                f'\'{num_classes_disc}\'.')
        model_num_classes = num_classes_gen or num_classes_disc

        if num_classes is not None and model_num_classes is not None:
            assert num_classes == model_num_classes, (
                'Input \'num_classes\' is inconsistent with '
                f'model\'s ones. Receive \'{num_classes}\' and '
                f'\'{model_num_classes}\'.')

        num_classes = num_classes or model_num_classes
        return num_classes

    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> List[DataSample]:
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
            List[DataSample]: Generated images or image dict.
        """
        if isinstance(inputs, Tensor):
            noise = inputs
            sample_kwargs = {}
        else:
            noise = inputs.get('noise', None)
            num_batches = get_valid_num_batches(inputs, data_samples)
            noise = self.noise_fn(noise, num_batches=num_batches)
            sample_kwargs = inputs.get('sample_kwargs', dict())
        num_batches = noise.shape[0]

        labels = self.data_sample_to_label(data_samples)
        if labels is None:
            labels = self.label_fn(num_batches=num_batches)

        sample_model = self._get_valid_model(inputs)
        batch_sample_list = []
        if sample_model in ['ema', 'orig']:
            if sample_model == 'ema':
                generator = self.generator_ema
            else:
                generator = self.generator
            outputs = generator(noise, label=labels, return_noise=False)
            outputs = self.data_preprocessor.destruct(outputs, data_samples)

            gen_sample = DataSample()
            if data_samples:
                gen_sample.update(data_samples)
            if isinstance(inputs, dict) and 'img' in inputs:
                gen_sample.gt_img = inputs['img']
            gen_sample.fake_img = outputs
            gen_sample.noise = noise
            gen_sample.set_gt_label(labels)
            gen_sample.sample_kwargs = deepcopy(sample_kwargs)
            gen_sample.sample_model = sample_model
            batch_sample_list = gen_sample.split(allow_nonseq_value=True)

        else:  # sample model in 'ema/orig'
            outputs_orig = self.generator(
                noise, label=labels, return_noise=False, **sample_kwargs)
            outputs_ema = self.generator_ema(
                noise, label=labels, return_noise=False, **sample_kwargs)
            outputs_orig = self.data_preprocessor.destruct(
                outputs_orig, data_samples)
            outputs_ema = self.data_preprocessor.destruct(
                outputs_ema, data_samples)

            gen_sample = DataSample()
            if data_samples:
                gen_sample.update(data_samples)
            if isinstance(inputs, dict) and 'img' in inputs:
                gen_sample.gt_img = inputs['img']
            gen_sample.ema = DataSample(fake_img=outputs_ema)
            gen_sample.orig = DataSample(fake_img=outputs_orig)
            gen_sample.noise = noise
            gen_sample.set_gt_label(labels)
            gen_sample.sample_kwargs = deepcopy(sample_kwargs)
            gen_sample.sample_model = 'ema/orig'
            batch_sample_list = gen_sample.split(allow_nonseq_value=True)

        return batch_sample_list

    def train_generator(self, inputs: dict, data_samples: List[DataSample],
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[DataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        num_batches = inputs['img'].shape[0]

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

    def train_discriminator(self, inputs: dict, data_samples: List[DataSample],
                            optimizer_wrapper: OptimWrapper
                            ) -> Dict[str, Tensor]:
        """Training function for discriminator. All GANs should implement this
        function by themselves.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[DataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        real_imgs = inputs['img']
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
