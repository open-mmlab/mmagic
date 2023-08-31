# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.hub import load_state_dict_from_url
from tqdm import tqdm
from transformers import CLIPTokenizer

from mmagic.registry import MODELS
from ..stable_diffusion import StableDiffusion
from .fastcomposer_util import FastComposerModel, get_object_transforms

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class FastComposer(StableDiffusion):

    def __init__(self,
                 pretrained_cfg: dict,
                 vae: ModelType,
                 text_encoder: ModelType,
                 tokenizer: str,
                 unet: ModelType,
                 scheduler: ModelType,
                 test_scheduler: Optional[ModelType] = None,
                 dtype: str = 'fp32',
                 enable_xformers: bool = True,
                 noise_offset_weight: float = 0,
                 tomesd_cfg: Optional[dict] = None,
                 data_preprocessor=dict(type='DataPreprocessor'),
                 init_cfg: Optional[dict] = None):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         test_scheduler, dtype, enable_xformers,
                         noise_offset_weight, tomesd_cfg, data_preprocessor,
                         init_cfg)
        self.vae = self.vae.model
        self.unet = self.unet.model
        model = FastComposerModel.from_pretrained(pretrained_cfg, self.vae,
                                                  self.unet)
        if pretrained_cfg['finetuned_model_path'][:7] == 'http://' or \
                pretrained_cfg['finetuned_model_path'][:8] == 'https://':
            model.load_state_dict(
                load_state_dict_from_url(
                    pretrained_cfg['finetuned_model_path'],
                    map_location='cpu'),
                strict=False)
        elif pretrained_cfg['finetuned_model_path']:
            model.load_state_dict(
                torch.load(
                    pretrained_cfg['finetuned_model_path'],
                    map_location='cpu'),
                strict=False)
        weight_dtype = torch.float32
        if dtype == 'fp16':
            weight_dtype = torch.float16
        elif dtype == 'bf16':
            weight_dtype = torch.bfloat16
        model = model.to(weight_dtype)

        self.unet = model.unet

        if pretrained_cfg['enable_xformers_memory_efficient_attention']:
            self.unet.enable_xformers_memory_efficient_attention()

        self.text_encoder = model.text_encoder
        self.image_encoder = model.image_encoder

        self.postfuse_module = model.postfuse_module

        del model

        self.special_tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_cfg['pretrained_model_name_or_path'],
            subfolder='tokenizer',
            revision=pretrained_cfg['revision'],
        )
        self.special_tokenizer.add_tokens(['img'], special_tokens=True)
        self.image_token_id = self.special_tokenizer.convert_tokens_to_ids(
            'img')

        self.object_transforms = get_object_transforms(pretrained_cfg)

    @torch.no_grad()
    def _tokenize_and_mask_noun_phrases_ends(self, caption):
        """Augment the text embedding."""

        input_ids = self.special_tokenizer.encode(caption)
        noun_phrase_end_mask = [False for _ in input_ids]
        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.special_tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [
                self.tokenizer.pad_token_id
            ] * (
                max_len - len(clean_input_ids))

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask))

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(
            noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    @torch.no_grad()
    def _encode_augmented_prompt(self, prompt: str,
                                 reference_images: List[Image.Image],
                                 device: torch.device,
                                 weight_dtype: torch.dtype):
        """Encode reference images.

        Args:
            prompt (str or list(int)): prompt to be encoded.
            reference_images: (List[Image.Image]): List of reference images.
            device (torch.device):torch device.
            weight_dtype (torch.dtype): torch.dtype.

        Returns:
            text_embeddings (torch.Tensor): text embeddings generated by
                clip text encoder.
        """
        # TODO: check this
        object_pixel_values = []
        for image in reference_images:
            image_tensor = torch.from_numpy(np.array(
                image.convert('RGB'))).permute(2, 0, 1)
            image = self.object_transforms(image_tensor)
            object_pixel_values.append(image)

        object_pixel_values = torch.stack(
            object_pixel_values,
            dim=0).to(memory_format=torch.contiguous_format).float()
        object_pixel_values = object_pixel_values.unsqueeze(0).to(
            dtype=weight_dtype, device=device)
        object_embeds = self.image_encoder(object_pixel_values)

        id_and_mask = self._tokenize_and_mask_noun_phrases_ends(prompt)
        input_ids, image_token_mask = id_and_mask
        input_ids, image_token_mask = input_ids.to(
            device), image_token_mask.to(device)

        num_objects = image_token_mask.sum(dim=1)

        augmented_prompt_embeds = self.postfuse_module(
            self.text_encoder(input_ids)[0], object_embeds, image_token_mask,
            num_objects)
        return augmented_prompt_embeds

    @torch.no_grad()
    def infer(self,
              prompt: Union[str, List[str]] = None,
              height: Optional[int] = None,
              width: Optional[int] = None,
              num_inference_steps: int = 50,
              guidance_scale: float = 7.5,
              negative_prompt: Optional[Union[str, List[str]]] = None,
              num_images_per_prompt: Optional[int] = 1,
              eta: float = 0.0,
              generator: Optional[Union[torch.Generator,
                                        List[torch.Generator]]] = None,
              latents: Optional[torch.FloatTensor] = None,
              prompt_embeds: Optional[torch.FloatTensor] = None,
              negative_prompt_embeds: Optional[torch.FloatTensor] = None,
              output_type: Optional[str] = 'pil',
              return_dict: bool = True,
              callback: Optional[Callable[[int, int, torch.FloatTensor],
                                          None]] = None,
              callback_steps: int = 1,
              cross_attention_kwargs: Optional[Dict[str, Any]] = None,
              alpha_: float = 0.7,
              reference_subject_images: List[Image.Image] = None,
              augmented_prompt_embeds: Optional[torch.FloatTensor] = None,
              show_progress: bool = True):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation.
                If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*):
                defaults to
                self.unet.config.sample_size * self.vae_scale_factor
                The height in pixels of the generated image.
            width (`int`, *optional*):
                defaults to
                self.unet.config.sample_size * self.vae_scale_factor
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
                More denoising steps usually lead to a higher quality image
                at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in
                [Classifier-Free Diffusion Guidance]
                (https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation
                2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
                Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images
                that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
                If not defined, one has to pass
                `negative_prompt_embeds` instead.
                Ignored when not using guidance
                (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper:
                https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`,
             *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/
                docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents,
                sampled from a Gaussian distribution,
                to be used as inputs for image generation.
                Can be used to tweak the same generation with
                different prompts. If not provided, a latents tensor
                will ge generated by sampling using
                the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings.
                Can be used to easily tweak text inputs,
                *e.g.* prompt weighting. If not provided,
                text embeddings will be generated from `prompt`
                input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings.
                Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be
                generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/):
                `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]
                instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps`
                steps during inference. The function will be
                called with the following arguments: `callback(step: int,
                timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called.
                If not specified, the callback will be called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the
                `AttentionProcessor` as defined under `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/
                diffusers/blob/main/src/diffusers/models/cross_attention.py).
            alpha_ (`float`, defaults to 0.7):
                The ratio of subject conditioning. If `alpha_` is 0.7,
                the beginning 30% of denoising steps use text prompts,
                while the last 70% utilize image-augmented prompts.
                Increase alpha for identity preservation,
                decrease it for prompt consistency.
            reference_subject_images (`List[PIL.Image.Image]`):
                a list of PIL images that are used as reference subjects.
                The number of images should be equal to the
                number of augmented tokens in the prompts.
            augmented_prompt_embeds: (`torch.FloatTensor`, *optional*):
                Pre-generated image augmented text embeddings.
                If not provided, embeddings will be generated from `prompt`
                and `reference_subject_images`.
            show_progress: ('bool'):
                show progress or not.
        Examples:

        Returns:
            `OrderedDict` or `tuple`:
            `OrderedDict` if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with
            the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding
            generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
        )

        assert_text = 'Prompt and reference subject images or prompt_embeds ' \
                      'and augmented_prompt_embeds must be provided.'

        assert (prompt and reference_subject_images) or \
               (prompt_embeds and augmented_prompt_embeds), assert_text

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device
        # here `guidance_scale` is defined analog to
        # the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
        # `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        assert do_classifier_free_guidance

        # 3. Encode input prompt
        prompt_text_only = prompt.replace('img', '')

        prompt_embeds = self._encode_prompt(prompt_text_only, device,
                                            num_images_per_prompt,
                                            do_classifier_free_guidance,
                                            negative_prompt)

        if augmented_prompt_embeds is None:
            augmented_prompt_embeds = self._encode_augmented_prompt(
                prompt, reference_subject_images, device, prompt_embeds.dtype)
            augmented_prompt_embeds = augmented_prompt_embeds.repeat(
                num_images_per_prompt, 1, 1)

        prompt_embeds = torch.cat([prompt_embeds, augmented_prompt_embeds],
                                  dim=0)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        start_subject_conditioning_step = (1 - alpha_) * num_inference_steps

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        (null_prompt_embeds, text_prompt_embeds,
         augmented_prompt_embeds) = prompt_embeds.chunk(3)

        # 7. Denoising loop
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order
        if show_progress:
            timesteps = tqdm(timesteps)
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] *
                          2) if do_classifier_free_guidance else latents)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            if i <= start_subject_conditioning_step:
                current_prompt_embeds = torch.cat(
                    [null_prompt_embeds, text_prompt_embeds], dim=0)
            else:
                current_prompt_embeds = torch.cat(
                    [null_prompt_embeds, augmented_prompt_embeds], dim=0)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=current_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            else:
                assert 0, 'Not Implemented'

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or (i + 1 > num_warmup_steps and
                                           (i + 1) % self.scheduler.order
                                           == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
        has_nsfw_concept = None
        if output_type == 'latent':
            image = latents
            has_nsfw_concept = None
        elif output_type == 'pil':
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 10. Convert to PIL
            image = self.output_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return OrderedDict(
            samples=image, nsfw_content_detected=has_nsfw_concept)
