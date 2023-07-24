# Copyright (c) OpenMMLab. All rights reserved.
from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log
from mmengine.model import is_model_wrapper
from mmengine.optim import OptimWrapperDict
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from mmagic.models.archs import AttentionInjection
from mmagic.models.utils import build_module
from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList
from ..stable_diffusion import StableDiffusion
from .controlnet_utils import change_base_model

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class ControlStableDiffusion(StableDiffusion):
    """Implementation of `ControlNet with Stable Diffusion.

    <https://arxiv.org/abs/2302.05543>`_ (ControlNet).

    Args:
        vae (Union[dict, nn.Module]): The config or module for VAE model.
        text_encoder (Union[dict, nn.Module]): The config or module for text
            encoder.
        tokenizer (str): The **name** for CLIP tokenizer.
        unet (Union[dict, nn.Module]): The config or module for Unet model.
        controlnet (Union[dict, nn.Module]): The config or module for
            ControlNet.
        schedule (Union[dict, nn.Module]): The config or module for diffusion
            scheduler.
        test_scheduler (Union[dict, nn.Module], optional): The config or
            module for diffusion scheduler in test stage (`self.infer`). If not
            passed, will use the same scheduler as `schedule`. Defaults to
            None.
        dtype (str, optional): The dtype for the model. Defaults to 'fp16'.
        enable_xformers (bool, optional): Whether to use xformers.
            Defaults to True.
        noise_offset_weight (bool, optional): The weight of noise offset
            introduced in https://www.crosslabs.org/blog/diffusion-with-offset-noise  # noqa
            Defaults to 0.
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
                 controlnet: ModelType,
                 scheduler: ModelType,
                 test_scheduler: Optional[ModelType] = None,
                 dtype: str = 'fp32',
                 enable_xformers: bool = True,
                 noise_offset_weight: float = 0,
                 tomesd_cfg: Optional[dict] = None,
                 data_preprocessor=dict(type='DataPreprocessor'),
                 init_cfg: Optional[dict] = None,
                 attention_injection=False):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         test_scheduler, dtype, enable_xformers,
                         noise_offset_weight, tomesd_cfg, data_preprocessor,
                         init_cfg)

        default_args = dict()
        if dtype is not None:
            default_args['dtype'] = dtype

        # NOTE: initialize controlnet as fp32
        self.controlnet = build_module(controlnet, MODELS)
        self._controlnet_ori_dtype = next(self.controlnet.parameters()).dtype
        print_log(
            'Set ControlNetModel dtype to '
            f'\'{self._controlnet_ori_dtype}\'.', 'current')
        self.set_xformers(self.controlnet)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        if attention_injection:
            self.unet = AttentionInjection(self.unet)

    def init_weights(self):
        """Initialize the weights. Noted that this function will only be called
        at train. If you want to inference with a different unet model, you can
        call this function manually or use
        `mmagic.models.editors.controlnet.controlnet_utils.change_base_model`
        to convert the weight of ControlNet manually.

        Example:
        >>> 1. init controlnet from unet
        >>> init_cfg = dict(type='init_from_unet')

        >>> 2. switch controlnet weight from unet
        >>> # base model is not defined, use `runwayml/stable-diffusion-v1-5`
        >>> # as default
        >>> init_cfg = dict(type='convert_from_unet')
        >>> # base model is defined
        >>> init_cfg = dict(
        >>>     type='convert_from_unet',
        >>>     base_model=dict(
        >>>         type='UNet2DConditionModel',
        >>>         from_pretrained='REPO_ID',
        >>>         subfolder='unet'))
        """
        if self.init_cfg is not None:
            init_type = self.init_cfg.get('type', None)
        else:
            init_type = None

        if init_type == 'init_from_unet':
            # fetch module
            if is_model_wrapper(self.controlnet):
                controlnet = self.controlnet.module
            else:
                controlnet = self.controlnet

            if is_model_wrapper(self.unet):
                unet = self.unet.module
            else:
                unet = self.unet

            if controlnet._from_pretrained is not None:
                print_log(
                    'ControlNet has initialized from pretrained '
                    f'weight \'{controlnet._from_pretrained}\'.'
                    ' Re-initialize ControlNet from Unet.', 'current', WARNING)

            # copy weight
            log_template = 'Initialize weight ControlNet from Unet: {}'
            for n, p in unet.named_parameters():
                if n in controlnet.state_dict():
                    print_log(log_template.format(n), 'current')
                    controlnet.state_dict()[n].copy_(p.data)

            # check zero_conv
            zero_conv_blocks = controlnet.controlnet_down_blocks
            for n, p in zero_conv_blocks.named_parameters():
                if not (p == 0).all():
                    print_log(f'{n} in ControlNet is not initialized with '
                              'zero. Set to zero manually.')
                    p.data.zero_()

        elif init_type == 'convert_from_unet':
            # fetch module
            if is_model_wrapper(self.controlnet):
                controlnet = self.controlnet.module
            else:
                controlnet = self.controlnet

            if is_model_wrapper(self.unet):
                unet = self.unet.module
            else:
                unet = self.unet

            # use sd-v15 as base model by default
            base_model_default_cfg = dict(
                type='UNet2DConditionModel',
                from_pretrained='runwayml/stable-diffusion-v1-5',
                subfolder='unet')
            base_model_cfg = self.init_cfg.get('base_model',
                                               base_model_default_cfg)
            base_model = MODELS.build(base_model_cfg)
            change_base_model(controlnet, unet, base_model)

        else:
            assert init_type is None, (
                'Only support \'init_from_unet\', \'convert_from_unet\' or '
                f'None. But receive {init_type}.')

    def train_step(self, data: dict,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        """Train step for ControlNet model.
        Args:
            data (dict): Data sampled from dataloader.
            optim_wrapper (OptimWrapperDict): OptimWrapperDict instance
                contains OptimWrapper of generator and discriminator.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        data = self.data_preprocessor(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        optimizer = optim_wrapper['controlnet']

        with optimizer.optim_context(self.controlnet):
            target = inputs['target']
            control = (inputs['source'] + 1) / 2  # [-1, 1] -> [0, 1]
            prompt = data_samples.prompt

            num_batches = target.shape[0]

            target = target.to(self.dtype)
            latents = self.vae.encode(target).latent_dist.sample()
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

            # forward control
            # NOTE: we train controlnet in fp32, convert to float manually
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents.float(),
                timesteps,
                encoder_hidden_states=encoder_hidden_states.float(),
                controlnet_cond=control.float(),
                return_dict=False,
            )
            # Predict the noise residual and compute loss
            # NOTE: we train unet in fp32, convert to float manually
            model_output = self.unet(
                noisy_latents.float(),
                timesteps,
                encoder_hidden_states=encoder_hidden_states.float(),
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample)
            model_pred = model_output['sample']

            loss = F.mse_loss(model_pred.float(), gt.float(), reduction='mean')

            optimizer.update_params(loss)

        return dict(loss=loss)

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
        prompt = data['data_samples'].prompt
        control = data['inputs']['source']

        output = self.infer(
            prompt, control=((control + 1) / 2), return_type='tensor')
        samples = output['samples']

        samples = self.data_preprocessor.destruct(
            samples, data['data_samples'], key='target')
        control = self.data_preprocessor.destruct(
            control, data['data_samples'], key='source')

        data_sample = DataSample(
            fake_img=samples,
            control=control,
            prompt=data['data_samples'].prompt)
        data_sample_list = data_sample.split()
        return data_sample_list

    def test_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Calls
        ``self.data_preprocessor`` and ``self.infer`` in order. Return the
        generated results which will be passed to evaluator or visualizer.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            SampleList: Generated image or image dict.
        """
        data = self.data_preprocessor(data)
        prompt = data['data_samples'].prompt
        control = data['inputs']['source']

        output = self.infer(
            prompt, control=((control + 1) / 2), return_type='tensor')
        samples = output['samples']

        samples = self.data_preprocessor.destruct(
            samples, data['data_samples'], key='target')
        control = self.data_preprocessor.destruct(
            control, data['data_samples'], key='source')

        data_sample = DataSample(
            fake_img=samples,
            control=control,
            prompt=data['data_samples'].prompt)
        data_sample_list = data_sample.split()
        return data_sample_list

    # NOTE: maybe we should do this in a controlnet preprocessor
    @staticmethod
    def prepare_control(image: Tuple[Image.Image, List[Image.Image], Tensor,
                                     List[Tensor]], width: int, height: int,
                        batch_size: int, num_images_per_prompt: int,
                        device: str, dtype: str) -> Tensor:
        """A helper function to prepare single control images.

        Args:
            image (Tuple[Image.Image, List[Image.Image], Tensor, List[Tensor]]):  # noqa
                The input image for control.
            batch_size (int): The number of the prompt. The control will
                be repeated for `batch_size` times.
            num_images_per_prompt (int): The number images generate for one
                prompt.
            device (str): The device of the control.
            dtype (str): The dtype of the control.

        Returns:
            Tensor: The control in torch.tensor.
        """
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image[0], Image.Image):
                image = [
                    img.resize((width, height), resample=Image.LANCZOS)
                    for img in image
                ]
                image = [np.array(img)[None, :] for img in image]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size * num_images_per_prompt
        else:
            assert image_batch_size == batch_size, (
                'The number of Control condition must be 1 or equal to the '
                'number of prompt.')
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    def train(self, mode: bool = True):
        """Set train/eval mode.

        Args:
            mode (bool, optional): Whether set train mode. Defaults to True.
        """
        if mode:
            if next(self.controlnet.parameters()
                    ).dtype != self._controlnet_ori_dtype:
                print_log(
                    'Set ControlNetModel dtype to '
                    f'\'{self._controlnet_ori_dtype}\' in the train mode.',
                    'current')
            self.controlnet.to(self._controlnet_ori_dtype)
        else:
            self.controlnet.to(self.dtype)
            print_log(
                f'Set ControlNetModel dtype to \'{self.dtype}\' '
                'in the eval mode.', 'current')
        return super().train(mode)

    @torch.no_grad()
    def infer(self,
              prompt: Union[str, List[str]],
              height: Optional[int] = None,
              width: Optional[int] = None,
              control: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
              controlnet_conditioning_scale: float = 1.0,
              num_inference_steps: int = 20,
              guidance_scale: float = 7.5,
              negative_prompt: Optional[Union[str, List[str]]] = None,
              num_images_per_prompt: Optional[int] = 1,
              eta: float = 0.0,
              generator: Optional[torch.Generator] = None,
              latents: Optional[torch.FloatTensor] = None,
              return_type='image',
              show_progress=True):
        """Function invoked when calling the pipeline for generation.

        Args:
            prompt (str or List[str]): The prompt or prompts to guide
                the image generation.
            height (int, Optional): The height in pixels of the generated
                image. If not passed, the height will be
                `self.unet_sample_size * self.vae_scale_factor` Defaults
                to None.
            width (int, Optional): The width in pixels of the generated image.
                If not passed, the width will be
                `self.unet_sample_size * self.vae_scale_factor` Defaults
                to None.
            num_inference_steps (int): The number of denoising steps.
                More denoising steps usually lead to a higher quality image at
                the expense of slower inference. Defaults to 50.
            guidance_scale (float): Guidance scale as defined in Classifier-
                Free Diffusion Guidance (https://arxiv.org/abs/2207.12598).
                Defaults to 7.5
            negative_prompt (str or List[str], optional): The prompt or
                prompts not to guide the image generation. Ignored when not
                using guidance (i.e., ignored if `guidance_scale` is less
                than 1). Defaults to None.
            num_images_per_prompt (int): The number of images to generate
                per prompt. Defaults to 1.
            eta (float): Corresponds to parameter eta (η) in the DDIM paper:
                https://arxiv.org/abs/2010.02502. Only applies to
                DDIMScheduler, will be ignored for others. Defaults to 0.0.
            generator (torch.Generator, optional): A torch generator to make
                generation deterministic. Defaults to None.
            latents (torch.FloatTensor, optional): Pre-generated noisy latents,
                sampled from a Gaussian distribution, to be used as inputs for
                image generation. Can be used to tweak the same generation with
                different prompts. If not provided, a latents tensor will be
                generated by sampling using the supplied random `generator`.
                Defaults to None.
            return_type (str): The return type of the inference results.
                Supported types are 'image', 'numpy', 'tensor'. If 'image'
                is passed, a list of PIL images will be returned. If 'numpy'
                is passed, a numpy array with shape [N, C, H, W] will be
                returned, and the value range will be same as decoder's
                output range. If 'tensor' is passed, the decoder's output
                will be returned. Defaults to 'image'.

        Returns:
            dict: A dict containing the generated images and Control image.
        """
        assert return_type in ['image', 'tensor', 'numpy']

        # 0. Default height and width to unet
        height = height or self.unet_sample_size * self.vae_scale_factor
        width = width or self.unet_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self.device
        # here `guidance_scale` is defined analog to the
        # guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
        # `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        img_dtype = self.vae.module.dtype if hasattr(self.vae, 'module') \
            else self.vae.dtype
        if is_model_wrapper(self.controlnet):
            control_dtype = self.controlnet.module.dtype
        else:
            control_dtype = self.controlnet.dtype
        controls = self.prepare_control(
            control,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype=control_dtype)
        if do_classifier_free_guidance:
            controls = torch.cat([controls] * 2)

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, device,
                                              num_images_per_prompt,
                                              do_classifier_free_guidance,
                                              negative_prompt)

        # 4. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.test_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.test_scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        if show_progress:
            timesteps = tqdm(timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.test_scheduler.scale_model_input(
                latent_model_input, t)

            latent_model_input = latent_model_input.to(control_dtype)
            text_embeddings = text_embeddings.to(control_dtype)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=controls,
                return_dict=False,
            )

            down_block_res_samples = [
                down_block_res_sample * controlnet_conditioning_scale
                for down_block_res_sample in down_block_res_samples
            ]
            mid_block_res_sample *= controlnet_conditioning_scale

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )['sample']

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.test_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs)['prev_sample']

        # 8. Post-processing
        image = self.decode_latents(latents.to(img_dtype))

        if do_classifier_free_guidance:
            controls = torch.split(controls, controls.shape[0] // 2, dim=0)[0]

        if return_type == 'image':
            image = self.output_to_pil(image)
            controls = self.output_to_pil(controls * 2 - 1)
        elif return_type == 'numpy':
            image = image.cpu().numpy()
            controls = controls.cpu().numpy()
        else:
            assert return_type == 'tensor', (
                'Only support \'image\', \'numpy\' and \'tensor\' for '
                f'return_type, but receive {return_type}')

        return {'samples': image, 'controls': controls}

    def forward(self, *args, **kwargs):
        """forward is not implemented now."""
        raise NotImplementedError(
            'Forward is not implemented now, please use infer.')


@MODELS.register_module()
class ControlStableDiffusionImg2Img(ControlStableDiffusion):

    def _default_height_width(self, height, width, image):
        if isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[3]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[2]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(
            int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.test_scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self,
                        image,
                        timestep,
                        batch_size,
                        num_images_per_prompt,
                        dtype,
                        device,
                        generator=None,
                        noise=None):
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(
                f'`image` has to be of type `torch.Tensor`, '
                f' `PIL.Image.Image` or list but is {type(image)}')

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of '
                f' length {len(generator)}, but requested an effective batch'
                f' size of {batch_size}. Make sure the batch size '
                f' matches the length of the generators.')

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i:i +
                                      1]).latent_dist.sample(generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        vae_encode_latents = init_latents

        if batch_size > init_latents.shape[0] and \
                batch_size % init_latents.shape[0] == 0:
            raise ValueError(
                f'Cannot duplicate `image` of batch size'
                f' {init_latents.shape[0]} to {batch_size} text prompts.')
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        if noise is None:
            noise = torch.randn(
                shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        return init_latents, vae_encode_latents

    def prepare_latent_image(self, image, dtype):
        if isinstance(image, torch.Tensor):
            # Batch single image
            if image.ndim == 3:
                image = image.unsqueeze(0)

            image = image.to(dtype=dtype)
        else:
            # preprocess image
            if isinstance(image, (Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], Image.Image):
                image = [np.array(i.convert('RGB'))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=dtype) / 127.5 - 1.0

        return image

    @torch.no_grad()
    def infer(
        self,
        prompt: Union[str, List[str]],
        latent_image: Union[torch.FloatTensor, Image.Image,
                            List[torch.FloatTensor], List[Image.Image]] = None,
        latent_mask: torch.FloatTensor = None,
        strength: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        control: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        controlnet_conditioning_scale: float = 1.0,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        return_type='image',
        show_progress=True,
        reference_img: Union[torch.FloatTensor, Image.Image,
                             List[torch.FloatTensor],
                             List[Image.Image]] = None,
    ):
        """Function invoked when calling the pipeline for generation.

        Args:
            prompt (str or List[str]): The prompt or prompts to guide
                the image generation.
            height (int, Optional): The height in pixels of the generated
                image. If not passed, the height will be
                `self.unet_sample_size * self.vae_scale_factor` Defaults
                to None.
            width (int, Optional): The width in pixels of the generated image.
                If not passed, the width will be
                `self.unet_sample_size * self.vae_scale_factor` Defaults
                to None.
            num_inference_steps (int): The number of denoising steps.
                More denoising steps usually lead to a higher quality image at
                the expense of slower inference. Defaults to 50.
            guidance_scale (float): Guidance scale as defined in Classifier-
                Free Diffusion Guidance (https://arxiv.org/abs/2207.12598).
                Defaults to 7.5
            negative_prompt (str or List[str], optional): The prompt or
                prompts not to guide the image generation. Ignored when not
                using guidance (i.e., ignored if `guidance_scale` is less
                than 1). Defaults to None.
            num_images_per_prompt (int): The number of images to generate
                per prompt. Defaults to 1.
            eta (float): Corresponds to parameter eta (η) in the DDIM paper:
                https://arxiv.org/abs/2010.02502. Only applies to
                DDIMScheduler, will be ignored for others. Defaults to 0.0.
            generator (torch.Generator, optional): A torch generator to make
                generation deterministic. Defaults to None.
            latents (torch.FloatTensor, optional): Pre-generated noisy latents,
                sampled from a Gaussian distribution, to be used as inputs for
                image generation. Can be used to tweak the same generation with
                different prompts. If not provided, a latents tensor will be
                generated by sampling using the supplied random `generator`.
                Defaults to None.
            return_type (str): The return type of the inference results.
                Supported types are 'image', 'numpy', 'tensor'. If 'image'
                is passed, a list of PIL images will be returned. If 'numpy'
                is passed, a numpy array with shape [N, C, H, W] will be
                returned, and the value range will be same as decoder's
                output range. If 'tensor' is passed, the decoder's output
                will be returned. Defaults to 'image'.

        Returns:
            dict: A dict containing the generated images and Control image.
        """
        assert return_type in ['image', 'tensor', 'numpy']

        # 0. Default height and width to unet
        # height = height or self.unet_sample_size * self.vae_scale_factor
        # width = width or self.unet_sample_size * self.vae_scale_factor

        height, width = self._default_height_width(height, width, control)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self.device
        # here `guidance_scale` is defined analog to the
        # guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
        # `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        img_dtype = self.vae.module.dtype if hasattr(
            self.vae, 'module') else self.vae.dtype
        if is_model_wrapper(self.controlnet):
            control_dtype = self.controlnet.module.dtype
        else:
            control_dtype = self.controlnet.dtype
        controls = self.prepare_control(
            control,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype=control_dtype)
        if do_classifier_free_guidance:
            controls = torch.cat([controls] * 2)

        latent_image = self.prepare_latent_image(latent_image,
                                                 self.controlnet.dtype)

        if reference_img is not None:
            reference_img = self.prepare_latent_image(reference_img,
                                                      self.controlnet.dtype)

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, device,
                                              num_images_per_prompt,
                                              do_classifier_free_guidance,
                                              negative_prompt)
        text_embeddings = text_embeddings.to(control_dtype)

        # 4. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.test_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.test_scheduler.timesteps

        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device)

        latent_timestep = timesteps[:1].repeat(batch_size *
                                               num_images_per_prompt)

        # 5. Prepare latent variables
        latents, vae_encode_latents = self.prepare_latents(
            latent_image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            text_embeddings.dtype,
            device,
            generator,
            noise=latents)

        if reference_img is not None:
            _, ref_img_vae_latents = self.prepare_latents(
                reference_img,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                text_embeddings.dtype,
                device,
                generator,
                noise=latents)

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_test_scheduler_extra_step_kwargs(
            generator, eta)

        # 7. Denoising loop
        if show_progress:
            timesteps = tqdm(timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.test_scheduler.scale_model_input(
                latent_model_input, t)

            latent_model_input = latent_model_input.to(control_dtype)

            if reference_img is not None:
                ref_img_vae_latents_t = self.scheduler.add_noise(
                    ref_img_vae_latents, torch.randn_like(ref_img_vae_latents),
                    t)
                ref_img_vae_latents_model_input = torch.cat(
                    [ref_img_vae_latents_t] * 2) if \
                    do_classifier_free_guidance else ref_img_vae_latents_t
                ref_img_vae_latents_model_input =  \
                    self.test_scheduler.scale_model_input(
                        ref_img_vae_latents_model_input, t)
                ref_img_vae_latents_model_input = \
                    ref_img_vae_latents_model_input.to(control_dtype)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=controls,
                return_dict=False,
            )

            down_block_res_samples = [
                down_block_res_sample * controlnet_conditioning_scale
                for down_block_res_sample in down_block_res_samples
            ]
            mid_block_res_sample *= controlnet_conditioning_scale

            # predict the noise residual
            if reference_img is not None:
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    ref_x=ref_img_vae_latents_model_input)['sample']
            else:
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )['sample']

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.test_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs)['prev_sample']

        if latent_mask is not None:
            latents = latents * latent_mask + \
                vae_encode_latents * (1.0 - latent_mask)

        # 8. Post-processing
        image = self.decode_latents(latents.to(img_dtype))

        if do_classifier_free_guidance:
            controls = torch.split(controls, controls.shape[0] // 2, dim=0)[0]

        if return_type == 'image':
            image = self.output_to_pil(image)
            controls = self.output_to_pil(controls * 2 - 1)
        elif return_type == 'numpy':
            image = image.cpu().numpy()
            controls = controls.cpu().numpy()
        else:
            assert return_type == 'tensor', (
                'Only support \'image\', \'numpy\' and \'tensor\' for '
                f'return_type, but receive {return_type}')

        return {'samples': image, 'controls': controls}
