# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.runner import set_random_seed
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

from mmedit.models.utils import build_module, set_xformers
from mmedit.registry import DIFFUSION_SCHEDULERS, MODELS

logger = MMLogger.get_current_instance()

ModelType = Union[Dict, nn.Module]


@MODELS.register_module('sd')
@MODELS.register_module()
class StableDiffusion(BaseModel):
    """Class for Stable Diffusion. Refers to https://github.com/Stability-
    AI/stablediffusion and https://github.com/huggingface/diffusers/blob/main/s
    rc/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_attend_an
    d_excite.py  # noqa.

    Args:
        unet (Union[dict, nn.Module]): The config or module for Unet model.
        text_encoder (Union[dict, nn.Module]): The config or module for text
            encoder.
        vae (Union[dict, nn.Module]): The config or module for VAE model.
        tokenizer (str): The **name** for CLIP tokenizer.
        schedule (Union[dict, nn.Module]): The config or module for diffusion
            scheduler.
        test_scheduler (Union[dict, nn.Module], optional): The config or
            module for diffusion scheduler in test stage (`self.infer`). If not
            passed, will use the same scheduler as `schedule`. Defaults to
            None.
        enable_xformers (bool, optional): Whether to use xformers.
            Defaults to True.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """

    def __init__(self,
                 vae: ModelType,
                 text_encoder: ModelType,
                 tokenizer: str,
                 unet: ModelType,
                 scheduler: ModelType,
                 test_scheduler: Optional[ModelType] = None,
                 enable_xformers: bool = True,
                 data_preprocessor: Optional[ModelType] = dict(
                     type='EditDataPreprocessor'),
                 init_cfg: Optional[dict] = None):

        # TODO: support `from_pretrained` for this class
        super().__init__(data_preprocessor, init_cfg)

        self.vae = build_module(vae, MODELS)
        self.unet = build_module(unet, MODELS)
        self.scheduler = build_module(scheduler, DIFFUSION_SCHEDULERS)
        if test_scheduler is None:
            self.test_scheduler = deepcopy(self.scheduler)
        else:
            self.test_scheduler = build_module(test_scheduler,
                                               DIFFUSION_SCHEDULERS)
        self.text_encoder = build_module(text_encoder, MODELS)
        if isinstance(tokenizer, nn.Module):
            self.tokenizer = tokenizer
        else:
            # NOTE: here we assume tokenizer is an string
            # TODO: maybe support a tokenizer wrapper later
            self.tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer, subfolder='tokenizer')

        self.unet_sample_size = self.unet.sample_size
        self.vae_scale_factor = 2**(len(self.vae.block_out_channels) - 1)

        self.enable_xformers = enable_xformers
        self.set_xformers()

    def set_xformers(self) -> nn.Module:
        """Set xformers for the model.

        Returns:
            nn.Module: The model with xformers.
        """
        if self.enable_xformers:
            set_xformers(self)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def infer(self,
              prompt: Union[str, List[str]],
              height: Optional[int] = None,
              width: Optional[int] = None,
              num_inference_steps: int = 50,
              guidance_scale: float = 7.5,
              negative_prompt: Optional[Union[str, List[str]]] = None,
              num_images_per_prompt: Optional[int] = 1,
              eta: float = 0.0,
              generator: Optional[torch.Generator] = None,
              latents: Optional[torch.FloatTensor] = None,
              show_progress=True,
              seed=1,
              return_type='image'):
        """Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*,
                defaults to self.unet_sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*,
                defaults to self.unet_sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
                More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in
                [Classifier-Free Diffusion Guidance]
                (https://arxiv.org/abs/2207.12598).
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
                Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper:
                https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator] to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents,
                sampled from a Gaussian distribution,
                to be used as inputs for image generation.
                Can be used to tweak the same generation
                with different prompts.
                If not provided, a latents tensor will be
                generated by sampling using the supplied random `generator`.
            return_type (str): The return type of the inference results.
                Supported types are 'image', 'numpy', 'tensor'. If 'image'
                is passed, a list of PIL images will be returned. If 'numpy'
                is passed, a numpy array with shape [N, C, H, W] will be
                returned, and the value range will be same as decoder's
                output range. If 'tensor' is passed, the decoder's output
                will be returned. Defaults to 'image'.

        Returns:
            dict: A dict containing the generated images.
        """
        assert return_type in ['image', 'tensor', 'numpy']
        set_random_seed(seed=seed)

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

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, device,
                                              num_images_per_prompt,
                                              do_classifier_free_guidance,
                                              negative_prompt)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

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
        # TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        if show_progress:
            timesteps = tqdm(timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs)['prev_sample']

        # 8. Post-processing
        image = self.decode_latents(latents.to(self.vae.dtype))
        if return_type == 'image':
            image = self.output_to_pil(image)
        elif return_type == 'numpy':
            image = image.cpu().numpy()
        else:
            assert return_type == 'tensor', (
                'Only support \'image\', \'numpy\' and \'tensor\' for '
                f'return_type, but receive {return_type}')

        return {'samples': image}

    def output_to_pil(self, image) -> List[Image.Image]:
        """Convert output tensor to PIL image. Output tensor will be de-normed
        to [0, 255] by `EditDataPreprocessor.destruct`. Due to no
        `data_samples` is passed, color order conversion will not be performed.

        Args:
            image (torch.Tensor): The output tensor of the decoder.

        Returns:
            List[Image.Image]: The list of processed PIL images.
        """
        image = self.data_preprocessor.destruct(image)
        image = image.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        image = [Image.fromarray(img) for img in image]
        return image

    def _encode_prompt(self, prompt, device, num_images_per_prompt,
                       do_classifier_free_guidance, negative_prompt):
        """Encodes the prompt into text encoder hidden states.

        Args:
            prompt (str or list(int)): prompt to be encoded.
            device: (torch.device): torch device.
            num_images_per_prompt (int): number of images that should be
                generated per prompt.
            do_classifier_free_guidance (`bool`): whether to use classifier
                free guidance or not.
            negative_prompt (str or List[str]): The prompt or prompts not
                to guide the image generation. Ignored when not using
                guidance (i.e., ignored if `guidance_scale` is less than `1`).

        Returns:
            text_embeddings (torch.Tensor): text embeddings generated by
                clip text encoder.
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding='max_length', return_tensors='pt').input_ids

        if not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
            logger.warning(
                'The following part of your input was truncated because CLIP'
                ' can only handle sequences up to'
                f' {self.tokenizer.model_max_length} tokens: {removed_text}')

        if hasattr(self.text_encoder.config, 'use_attention_mask'
                   ) and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt,
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f'`negative_prompt` should be the same type to `prompt`,'
                    f'but got {type(negative_prompt)} !='
                    f' {type(prompt)}.')
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has '
                    f'batch size {len(negative_prompt)}, but `prompt`:'
                    f' {prompt} has batch size {batch_size}.'
                    f' Please make sure that passed `negative_prompt` matches'
                    ' the batch size of `prompt`.')
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding='max_length',
                max_length=max_length,
                truncation=True,
                return_tensors='pt',
            )

            if hasattr(self.text_encoder.config, 'use_attention_mask'
                       ) and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for
            # each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional
            # and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        """use vae to decode latents.

        Args:
            latents (torch.Tensor): latents to decode.

        Returns:
            image (torch.Tensor): image result.
        """
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']
        # we always cast to float32 as this does not cause
        # significant overhead and is compatible with bfloa16
        return image.float()

    def prepare_extra_step_kwargs(self, generator, eta):
        """prepare extra kwargs for the scheduler step.

        Args:
            generator (torch.Generator):
                generator for random functions.
            eta (float):
                eta (η) is only used with the DDIMScheduler,
                it will be ignored for other schedulers.
                eta corresponds to η in DDIM paper:
                https://arxiv.org/abs/2010.02502
                and should be between [0, 1]

        Return:
            extra_step_kwargs (dict):
                dict contains 'generator' and 'eta'
        """
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width):
        """check whether inputs are in suitable format or not."""

        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f'`prompt` has to be of '
                             f'type `str` or `list` but is {type(prompt)}')

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f'`height` and `width` have to be divisible '
                             f'by 8 but are {height} and {width}.')

    def prepare_latents(self,
                        batch_size,
                        num_channels_latents,
                        height,
                        width,
                        dtype,
                        device,
                        generator,
                        latents=None):
        """prepare latents for diffusion to run in latent space.

        Args:
            batch_size (int): batch size.
            num_channels_latents (int): latent channel nums.
            height (int): image height.
            width (int): image width.
            dtype (torch.dtype): float type.
            device (torch.device): torch device.
            generator (torch.Generator):
                generator for random functions, defaults to None.
            latents (torch.Tensor):
                Pre-generated noisy latents, defaults to None.

        Return:
            latents (torch.Tensor): prepared latents.
        """
        shape = (batch_size, num_channels_latents,
                 height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        if latents is None:
            latents = torch.randn(
                shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f'Unexpected latents shape, '
                                 f'got {latents.shape}, expected {shape}')
            latents = latents.to(device)

        # scale the initial noise by the standard
        # deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """forward is not implemented now."""
        raise NotImplementedError(
            'Forward is not implemented now, please use infer.')
