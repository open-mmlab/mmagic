# Copyright (c) OpenMMLab. All rights reserved.
import random
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log

from mmagic.models.archs import set_lora
from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList
from ..stable_diffusion.stable_diffusion import StableDiffusion

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class DreamBooth(StableDiffusion):
    """Implementation of `DreamBooth with Stable Diffusion.

    <https://arxiv.org/abs/2208.12242>`_ (DreamBooth).

    Args:
        vae (Union[dict, nn.Module]): The config or module for VAE model.
        text_encoder (Union[dict, nn.Module]): The config or module for text
            encoder.
        tokenizer (str): The **name** for CLIP tokenizer.
        unet (Union[dict, nn.Module]): The config or module for Unet model.
        schedule (Union[dict, nn.Module]): The config or module for diffusion
            scheduler.
        test_scheduler (Union[dict, nn.Module], optional): The config or
            module for diffusion scheduler in test stage (`self.infer`). If not
            passed, will use the same scheduler as `schedule`. Defaults to
            None.
        lora_config (dict, optional): The config for LoRA finetuning. Defaults
            to None.
        val_prompts (Union[str, List[str]], optional): The prompts for
            validation. Defaults to None.
        class_prior_prompt (str, optional): The prompt for class prior loss.
        num_class_images (int, optional): The number of images for class prior.
            Defaults to 3.
        prior_loss_weight (float, optional): The weight for class prior loss.
            Defaults to 0.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. Defaults to False.
        dtype (str, optional): The dtype for the model. Defaults to 'fp16'.
        enable_xformers (bool, optional): Whether to use xformers.
            Defaults to True.
        noise_offset_weight (bool, optional): The weight of noise offset
            introduced in https://www.crosslabs.org/blog/diffusion-with-offset-noise  # noqa
            Defaults to 0.
        tomesd_cfg (dict, optional): The config for TOMESD. Please refers to
            https://github.com/dbolya/tomesd and
            https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/utils/tome_utils.py for detail.  # noqa
            Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Defaults to
                dict(type='DataPreprocessor').
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`. Defaults to None/
    """

    def __init__(self,
                 vae: ModelType,
                 text_encoder: ModelType,
                 tokenizer: str,
                 unet: ModelType,
                 scheduler: ModelType,
                 test_scheduler: Optional[ModelType] = None,
                 lora_config: Optional[dict] = None,
                 val_prompts: Union[str, List[str]] = None,
                 class_prior_prompt: Optional[str] = None,
                 num_class_images: Optional[int] = 3,
                 prior_loss_weight: float = 0,
                 finetune_text_encoder: bool = False,
                 dtype: str = 'fp16',
                 enable_xformers: bool = True,
                 noise_offset_weight: float = 0,
                 tomesd_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[ModelType] = dict(
                     type='DataPreprocessor'),
                 init_cfg: Optional[dict] = None):

        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         test_scheduler, dtype, enable_xformers,
                         noise_offset_weight, tomesd_cfg, data_preprocessor,
                         init_cfg)
        self.num_class_images = num_class_images
        self.class_prior_prompt = class_prior_prompt
        self.prior_loss_weight = prior_loss_weight
        self.class_images = []

        self.dtype = torch.float32
        if dtype == 'fp16':
            self.dtype = torch.float16
        elif dtype == 'bf16':
            self.dtype = torch.bfloat16
        else:
            assert dtype in [
                'fp32', None
            ], ('dtype must be one of \'fp32\', \'fp16\', \'bf16\' or None.')

        self.finetune_text_encoder = finetune_text_encoder
        self.val_prompts = val_prompts
        self.lora_config = deepcopy(lora_config)

        self.prepare_model()
        self.set_lora()

    @torch.no_grad()
    def generate_class_prior_images(self, num_batches=None):
        """Generate images for class prior loss.

        Args:
            num_batches (int): Number of batches to generate images.
                If not passed, all images will be generated in one
                forward. Defaults to None.
        """
        if self.prior_loss_weight == 0:
            return
        if self.class_images:
            return

        assert self.class_prior_prompt is not None, (
            '\'class_prior_prompt\' must be set when \'prior_loss_weight\' is '
            'larger than 0.')
        assert self.num_class_images is not None, (
            '\'num_class_images\' must be set when \'prior_loss_weight\' is '
            'larger than 0.')

        print_log(
            'Generating class prior images with prompt: '
            f'{self.class_prior_prompt}', 'current')
        num_batches = num_batches or self.num_class_images

        unet_dtype = next(self.unet.parameters()).dtype
        self.unet.to(self.dtype)
        for idx in range(0, self.num_class_images, num_batches):
            prompt = self.class_prior_prompt
            if self.num_class_images > 1:
                prompt += f' {idx + 1} of {self.num_class_images}'

            output = self.infer(prompt, return_type='tensor')
            samples = output['samples']
            self.class_images.append(samples.clamp(-1, 1))
        self.unet.to(unet_dtype)

    def prepare_model(self):
        """Prepare model for training.

        Move model to target dtype and disable gradient for some models.
        """
        self.vae.requires_grad_(False)
        print_log('Set VAE untrainable.', 'current')
        self.vae.to(self.dtype)
        print_log(f'Move VAE to {self.dtype}.', 'current')
        if not self.finetune_text_encoder or self.lora_config:
            self.text_encoder.requires_grad_(False)
            print_log('Set Text Encoder untrainable.', 'current')
            self.text_encoder.to(self.dtype)
            print_log(f'Move Text Encoder to {self.dtype}.', 'current')
        if self.lora_config:
            self.unet.requires_grad_(False)
            print_log('Set Unet untrainable.', 'current')

    def set_lora(self):
        """Set LORA for model."""
        if self.lora_config:
            set_lora(self.unet, self.lora_config)

    @torch.no_grad()
    def val_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Calls
        ``self.data_preprocessor`` and ``self.infer`` in order. Return the
        generated results which will be passed to evaluator or visualizer.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            SampleList: Generated image or image dict.
        """
        data = self.data_preprocessor(data)
        data_samples = data['data_samples']
        if self.val_prompts is None:
            prompt = data_samples.prompt
        else:
            prompt = self.val_prompts
            # construct a fake data_sample for destruct
            data_samples.split() * len(prompt)
            data_samples = DataSample.stack(data_samples.split() * len(prompt))

        output = self.infer(prompt, return_type='tensor')
        samples = output['samples']
        samples = self.data_preprocessor.destruct(samples, data_samples)

        out_data_sample = DataSample(fake_img=samples, prompt=prompt)
        data_sample_list = out_data_sample.split()
        return data_sample_list

    @torch.no_grad()
    def test_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Calls
        ``self.data_preprocessor`` and ``self.infer`` in order. Return the
        generated results which will be passed to evaluator or visualizer.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            SampleList: Generated image or image dict.
        """
        if self.val_prompts is None:
            data = self.data_preprocessor(data)
            data_samples = data['data_samples']
            prompt = data_samples.prompt
        else:
            prompt = self.val_prompts
            # construct a fake data_sample for destruct
            data_samples = DataSample.stack(data['data_samples'] * len(prompt))

        output = self.infer(prompt, return_type='tensor')
        samples = output['samples']
        samples = self.data_preprocessor.destruct(samples, data_samples)

        out_data_sample = DataSample(fake_img=samples, prompt=prompt)
        data_sample_list = out_data_sample.split()
        return data_sample_list

    def train_step(self, data, optim_wrapper):

        data = self.data_preprocessor(data)
        inputs, data_samples = data['inputs'], data['data_samples']

        with optim_wrapper.optim_context(self.unet):
            image = inputs  # image for new concept
            prompt = data_samples.prompt
            num_batches = image.shape[0]

            if self.prior_loss_weight != 0:
                # image and prompt for prior preservation
                self.generate_class_prior_images(num_batches=num_batches)
                class_images_used = []
                for _ in range(num_batches):
                    idx = random.randint(0, len(self.class_images) - 1)
                    class_images_used.append(self.class_images[idx])

                image = torch.cat([image, *class_images_used], dim=0)
                prompt = prompt + [self.class_prior_prompt]

            image = image.to(self.dtype)
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                self.scheduler.num_train_timesteps, (num_batches, ),
                device=self.device)
            timesteps = timesteps.long()

            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            input_ids = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                return_tensors='pt',
                padding='max_length',
                truncation=True)['input_ids'].to(self.device)

            encoder_hidden_states = self.text_encoder(input_ids)[0]

            if self.scheduler.config.prediction_type == 'epsilon':
                gt = noise
            elif self.scheduler.config.prediction_type == 'v_prediction':
                gt = self.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError('Unknown prediction type '
                                 f'{self.scheduler.config.prediction_type}')

            # NOTE: we train unet in fp32, convert to float manually
            model_output = self.unet(
                noisy_latents.float(),
                timesteps,
                encoder_hidden_states=encoder_hidden_states.float())
            model_pred = model_output['sample']

            loss_dict = dict()
            if self.prior_loss_weight != 0:
                model_pred, prior_pred = model_pred.split(2, dim=1)
                gt, prior_gt = gt.split(2, dim=1)
                # calculate loss in FP32
                dreambooth_loss = F.mse_loss(model_pred.float(), gt.float())
                prior_loss = F.mse_loss(prior_pred.float(), prior_gt.float())
                loss_dict['dreambooth_loss'] = dreambooth_loss
                loss_dict['prior_loss'] = prior_loss * self.prior_loss_weight

            else:
                # calculate loss in FP32
                dreambooth_loss = F.mse_loss(model_pred.float(), gt.float())
                loss_dict['dreambooth_loss'] = dreambooth_loss

            parsed_loss, log_vars = self.parse_losses(loss_dict)
            optim_wrapper.update_params(parsed_loss)

        return log_vars

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """forward is not implemented now."""
        raise NotImplementedError(
            'Forward is not implemented now, please use infer.')
