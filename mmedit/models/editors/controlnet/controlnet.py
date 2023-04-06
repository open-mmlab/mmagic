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

from mmedit.models.utils import build_module
from mmedit.registry import MODELS
from mmedit.structures import EditDataSample
from mmedit.utils.typing import SampleList
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
        enable_xformers (bool, optional): Whether to use xformers.
            Defaults to True.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Defaults to
                dict(type='EditDataPreprocessor').
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
                 enable_xformers: bool = True,
                 data_preprocessor=dict(type='EditDataPreprocessor'),
                 init_cfg: Optional[dict] = None):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         test_scheduler, enable_xformers, data_preprocessor,
                         init_cfg)

        self.controlnet = build_module(controlnet, MODELS)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

    def init_weights(self):
        """Initialize the weights. Noted that this function will only be called
        at train. If you want to inference with a different unet model, you can
        call this function manually or use
        `mmedit.models.editors.controlnet.controlnet_utils.change_base_model`
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
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control,
                return_dict=False,
            )

            # Predict the noise residual and compute loss
            model_output = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
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

        data_sample = EditDataSample(
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

        data_sample = EditDataSample(
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
            batch_size (int): The batch size of the control. The control will
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
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    @torch.no_grad()
    def infer(self,
              prompt: Union[str, List[str]],
              height: Optional[int] = None,
              width: Optional[int] = None,
              control: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
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

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=controls,
                return_dict=False,
            )

            controlnet_conditioning_scale = 1.0
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
        image = self.decode_latents(latents)

        if return_type == 'image':
            image = self.output_to_pil(image)
        elif return_type == 'numpy':
            image = image.cpu().numpy()
        else:
            assert return_type == 'tensor', (
                'Only support \'image\', \'numpy\' and \'tensor\' for '
                f'return_type, but receive {return_type}')

        return {'samples': image, 'controls': controls}

    def forward(self, *args, **kwargs):
        """forward is not implemented now."""
        raise NotImplementedError(
            'Forward is not implemented now, please use infer.')
