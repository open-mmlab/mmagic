# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (FromSingleFileMixin, LoraLoaderMixin,
                               TextualInversionLoaderMixin)
from diffusers.models import (AsymmetricAutoencoderKL, AutoencoderKL,
                              UNet2DConditionModel)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (deprecate, is_accelerate_available,
                             is_accelerate_version, logging)
from diffusers.utils.torch_utils import randn_tensor
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def prepare_mask_and_masked_image(image,
                                  mask,
                                  height,
                                  width,
                                  return_image: bool = False):
    """Prepares a pair (image, mask) to be consumed by the Stable Diffusion
    pipeline. This means that those inputs will be converted to
    ``torch.Tensor`` with shapes ``batch x channels x height x width`` where
    ``channels`` is ``3`` for the ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized
    to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]):
            The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3``
            ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width``
            ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to
            inpaint.
            It can be a ``PIL.Image``, or a ``height x width``
            ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width``
            ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]``
        range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and
        ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (or the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as
            ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    if image is None:
        raise ValueError('`image` input cannot be undefined.')

    if mask is None:
        raise ValueError('`mask_image` input cannot be undefined.')

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError('`image` is a torch.Tensor but '
                            f'`mask` (type: {type(mask)} is not')

        # Batch single image
        if image.ndim == 3:
            assert image.shape[
                0] == 3, 'Image outside a batch should be of shape (3, H, W)'
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask
            # not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, \
            'Image and Mask must have 4 dimensions'
        assert image.shape[-2:] == mask.shape[
            -2:], 'Image and Mask must have the same spatial dimensions'
        assert image.shape[0] == mask.shape[
            0], 'Image and Mask must have the same batch size'

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError('Image should be in [-1, 1] range')

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError('Mask should be in [0, 1] range')

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(
            f'`mask` is a torch.Tensor but `image` (type: {type(image)} is not'
        )
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [
                i.resize((width, height), resample=PIL.Image.LANCZOS)
                for i in image
            ]
            image = [np.array(i.convert('RGB'))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [
                i.resize((width, height), resample=PIL.Image.LANCZOS)
                for i in mask
            ]
            mask = np.concatenate(
                [np.array(m.convert('L'))[None, None, :] for m in mask],
                axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image


class StableDiffusionInpaintPipeline(DiffusionPipeline,
                                     TextualInversionLoaderMixin,
                                     LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Pipeline for text-guided image inpainting using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the
        superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving,
        running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`]
            for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for
            loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for
            saving LoRA weights

    Args:
        vae ([`AutoencoderKL`, `AsymmetricAutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode
            images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14]
            (https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise
            the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images
            could be considered offensive or harmful.
            Please refer to the [model card]
            (https://huggingface.co/runwayml/stable-diffusion-v1-5)
            for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated
            images; used as inputs to the `safety_checker`.
    """
    _optional_components = ['safety_checker', 'feature_extractor']

    def __init__(
        self,
        vae: Union[AutoencoderKL, AsymmetricAutoencoderKL],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config,
                   'steps_offset') and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f'The configuration file of this scheduler: {scheduler} is '
                ' outdated. `steps_offset`'
                ' should be set to 1 instead of '
                f' {scheduler.config.steps_offset}. Please make sure '
                'to update the config accordingly as leaving `steps_offset` '
                ' might led to incorrect results'
                ' in future versions. If you have downloaded this checkpoint '
                ' from the Hugging Face Hub,'
                ' it would be very nice if you could open a Pull request for '
                ' the `scheduler/scheduler_config.json`'
                ' file')
            deprecate(
                'steps_offset!=1',
                '1.0.0',
                deprecation_message,
                standard_warn=False)
            new_config = dict(scheduler.config)
            new_config['steps_offset'] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(
                scheduler.config,
                'skip_prk_steps') and scheduler.config.skip_prk_steps is False:
            deprecation_message = (
                f'The configuration file of this scheduler: {scheduler} has '
                ' not set the configuration'
                ' `skip_prk_steps`. `skip_prk_steps` should be set to True in '
                ' the configuration file. Please make'
                ' sure to update the config accordingly as not setting '
                ' `skip_prk_steps` in the config might lead to'
                ' incorrect results in future versions. '
                ' If you have downloaded '
                ' this checkpoint from the Hugging Face'
                ' Hub, it would be very nice if you could open a Pull '
                ' request for the'
                ' `scheduler/scheduler_config.json` file')
            deprecate(
                'skip_prk_steps not set',
                '1.0.0',
                deprecation_message,
                standard_warn=False)
            new_config = dict(scheduler.config)
            new_config['skip_prk_steps'] = True
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f'You have disabled the safety checker for {self.__class__} '
                ' by passing `safety_checker=None`. Ensure'
                ' that you abide to the conditions of the Stable Diffusion '
                ' license and do not expose unfiltered'
                ' results in services or applications open to the public. '
                ' Both the diffusers team and Hugging Face'
                ' strongly recommend to keep the safety filter enabled in '
                ' all public facing circumstances, disabling'
                ' it only for use-cases that involve analyzing network '
                ' behavior or auditing its results. For more'
                ' information, please have a look at'
                ' https://github.com/huggingface/diffusers/pull/254 .')

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                'Make sure to define a feature extractor when '
                'loading {self.__class__} if you want to use the safety'
                ' checker. If you do not want to use the safety checker, '
                "you can pass `'safety_checker=None'` instead.")

        is_unet_version_less_0_9_0 = hasattr(
            unet.config, '_diffusers_version') and version.parse(
                version.parse(unet.config._diffusers_version).base_version
            ) < version.parse('0.9.0.dev0')
        is_unet_sample_size_less_64 = hasattr(
            unet.config, 'sample_size') and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                'The configuration file of the unet has set the default '
                '`sample_size` to smaller than'
                " 64 which seems highly unlikely .If you're checkpoint is "
                ' a fine-tuned version of any of the'
                ' following: \n- CompVis/stable-diffusion-v1-4 \n- '
                ' CompVis/stable-diffusion-v1-3 \n-'
                ' CompVis/stable-diffusion-v1-2 \n- '
                ' CompVis/stable-diffusion-v1-1 \n- '
                ' runwayml/stable-diffusion-v1-5'
                ' \n- runwayml/stable-diffusion-inpainting \n you should '
                " change 'sample_size' to 64 in the"
                ' configuration file. Please make sure to update the config '
                ' accordingly as leaving `sample_size=32`'
                ' in the config might lead to incorrect results in future '
                ' versions. If you have downloaded this'
                ' checkpoint from the Hugging Face Hub, it would be very nice '
                ' if you could open a Pull request for'
                ' the `unet/config.json` file')
            deprecate(
                'sample_size<64',
                '1.0.0',
                deprecation_message,
                standard_warn=False)
            new_config = dict(unet.config)
            new_config['sample_size'] = 64
            unet._internal_dict = FrozenDict(new_config)

        # Check shapes, assume num_channels_latents == 4,
        # num_channels_mask == 1, num_channels_masked == 4
        if unet.config.in_channels != 9:
            logger.info(
                f'You have loaded a UNet with {unet.config.in_channels} '
                'input channels which.')

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2**(
            len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(
            requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.
    # pipeline_stable_diffusion.
    # StableDiffusionPipeline.enable_model_cpu_offload
    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offload all models to CPU to reduce memory usage with a low impact
        on performance. Moves one whole model at a
        time to the GPU when its `forward` method is called, and the model
        remains in GPU until the next model runs.
        Memory savings are lower than using `enable_sequential_cpu_offload`,
        but performance is much better due to the
        iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(
                '>=', '0.17.0.dev0'):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError(
                '`enable_model_cpu_offload` requires `accelerate v0.17.0` '
                'or higher.')

        device = torch.device(f'cuda:{gpu_id}')

        if self.device.type != 'cpu':
            self.to('cpu', silence_dtype_warnings=True)
            torch.cuda.empty_cache()
            # otherwise we don't see the memory savings
            # (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(
                cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(
                self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_
    # stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        promptA,
        promptB,
        t,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_promptA=None,
        negative_promptB=None,
        t_nag=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
                If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using
                guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak
                text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt`
                input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily
                tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be
                generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the
                text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        prompt = promptA
        negative_prompt = negative_promptA

        if promptA is not None and isinstance(promptA, str):
            batch_size = 1
        elif promptA is not None and isinstance(promptA, list):
            batch_size = len(promptA)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                promptA = self.maybe_convert_prompt(promptA, self.tokenizer)

            text_inputsA = self.tokenizer(
                promptA,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            )
            text_inputsB = self.tokenizer(
                promptB,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            )
            text_input_idsA = text_inputsA.input_ids
            text_input_idsB = text_inputsB.input_ids
            untruncated_ids = self.tokenizer(
                promptA, padding='longest', return_tensors='pt').input_ids

            if untruncated_ids.shape[-1] >= text_input_idsA.shape[
                    -1] and not torch.equal(text_input_idsA, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
                logger.warning(
                    'The following part of your input was truncated because '
                    'CLIP can only handle sequences up to'
                    f' {self.tokenizer.model_max_length} '
                    f'tokens: {removed_text}')

            if hasattr(self.text_encoder.config, 'use_attention_mask'
                       ) and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputsA.attention_mask.to(device)
            else:
                attention_mask = None

            # print("text_input_idsA: ",text_input_idsA)
            # print("text_input_idsB: ",text_input_idsB)
            # print('t: ',t)

            prompt_embedsA = self.text_encoder(
                text_input_idsA.to(device),
                attention_mask=attention_mask,
            )
            prompt_embedsA = prompt_embedsA[0]

            prompt_embedsB = self.text_encoder(
                text_input_idsB.to(device),
                attention_mask=attention_mask,
            )
            prompt_embedsB = prompt_embedsB[0]
            prompt_embeds = prompt_embedsA * (t) + (1 - t) * prompt_embedsB
            # print("prompt_embeds: ",prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(
            dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt,
        # using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt,
                                           seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokensA: List[str]
            uncond_tokensB: List[str]
            if negative_prompt is None:
                uncond_tokensA = [''] * batch_size
                uncond_tokensB = [''] * batch_size
            elif prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f'`negative_prompt` should be the same type to `prompt`, '
                    ' but got {type(negative_prompt)} !='
                    f' {type(prompt)}.')
            elif isinstance(negative_prompt, str):
                uncond_tokensA = [negative_promptA]
                uncond_tokensB = [negative_promptB]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has batch '
                    f' size {len(negative_prompt)}, but `prompt`:'
                    f' {prompt} has batch size {batch_size}. Please make '
                    ' sure that passed `negative_prompt` matches'
                    ' the batch size of `prompt`.')
            else:
                uncond_tokensA = negative_promptA
                uncond_tokensB = negative_promptB

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokensA = self.maybe_convert_prompt(
                    uncond_tokensA, self.tokenizer)
                uncond_tokensB = self.maybe_convert_prompt(
                    uncond_tokensB, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_inputA = self.tokenizer(
                uncond_tokensA,
                padding='max_length',
                max_length=max_length,
                truncation=True,
                return_tensors='pt',
            )
            uncond_inputB = self.tokenizer(
                uncond_tokensB,
                padding='max_length',
                max_length=max_length,
                truncation=True,
                return_tensors='pt',
            )

            if hasattr(self.text_encoder.config, 'use_attention_mask'
                       ) and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_inputA.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embedsA = self.text_encoder(
                uncond_inputA.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embedsB = self.text_encoder(
                uncond_inputB.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embedsA[0] * (t_nag) + (
                1 - t_nag) * negative_prompt_embedsB[0]

            # negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per
            # prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings
            # into a single batch
            # to avoid doing two forward passes
            # print("prompt_embeds: ",prompt_embeds)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.
    # pipeline_stable_diffusion.
    # StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type='pil')
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(
                    image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors='pt').to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype))
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.
    # pipeline_stable_diffusion.
    # StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all
        # schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be
        # ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

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

    def check_inputs(
        self,
        prompt,
        height,
        width,
        strength,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(
                f'The value of strength should in [0.0, 1.0] but is {strength}'
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f'`height` and `width` have to be divisible by 8 '
                             f' but are {height} and {width}.')

        if (callback_steps is None) or (callback_steps is not None and
                                        (not isinstance(callback_steps, int)
                                         or callback_steps <= 0)):
            raise ValueError(
                f'`callback_steps` has to be a positive integer but '
                f' is {callback_steps} of type'
                f' {type(callback_steps)}.')

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: '
                ' {prompt_embeds}. Please make sure to'
                ' only forward one of the two.')
        elif prompt is None and prompt_embeds is None:
            raise ValueError('Provide either `prompt` or `prompt_embeds`. '
                             ' Cannot leave both '
                             ' `prompt` and `prompt_embeds` undefined.')
        elif prompt is not None and (not isinstance(prompt, str)
                                     and not isinstance(prompt, list)):
            raise ValueError(f'`prompt` has to be of type `str` or '
                             f' `list` but is {type(prompt)}')

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `negative_prompt`: {negative_prompt} '
                'and  `negative_prompt_embeds`:'
                f' {negative_prompt_embeds}. Please make sure to only '
                ' forward one of the two.')

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    '`prompt_embeds` and `negative_prompt_embeds` '
                    'must have the same shape when passed directly, but'
                    f' got: `prompt_embeds` {prompt_embeds.shape} != '
                    ' `negative_prompt_embeds`'
                    f' {negative_prompt_embeds.shape}.')

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (batch_size, num_channels_latents,
                 height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of '
                f' length {len(generator)},'
                ' but requested an effective batch'
                f' size of {batch_size}. Make sure the batch size '
                ' matches the length of the generators.')

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                'Since strength < 1. initial latents are to be initialised'
                ' as a combination of Image + Noise.'
                'However, either the image or the noise timestep '
                'has not been provided.')

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(
                image=image, generator=generator)

        if latents is None:
            noise = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise,
            # else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(
                image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the
            # Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma \
                if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents, )

        if return_noise:
            outputs += (noise, )

        if return_image_latents:
            outputs += (image_latents, )

        return outputs

    def _encode_vae_image(self, image: torch.Tensor,
                          generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i:i + 1]).latent_dist.sample(
                    generator=generator[i]) for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.sample(
                generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_mask_latents(self, mask, masked_image, batch_size, height,
                             width, dtype, device, generator,
                             do_classifier_free_guidance):
        # resize the mask to latents shape as we concatenate the
        # mask to the latents
        # we do that before converting to dtype to avoid breaking in
        # case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask,
            size=(height // self.vae_scale_factor,
                  width // self.vae_scale_factor))
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)
        masked_image_latents = self._encode_vae_image(
            masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per
        # prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. "
                    ' Masks are supposed to be duplicated to'
                    f' a total batch size of {batch_size}, '
                    f' but {mask.shape[0]} '
                    ' masks were passed. Make sure the number'
                    ' of masks that you pass is divisible by the '
                    ' total requested batch size.')
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't "
                    ' match. Images are supposed to be duplicated'
                    f' to a total batch size of {batch_size}, '
                    f' but {masked_image_latents.shape[0]} images were passed.'
                    ' Make sure the number of images that you pass is '
                    ' divisible by the total requested batch size.')
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2)
            if do_classifier_free_guidance else masked_image_latents)

        # aligning device to prevent device errors when concatenating it
        # with the latent model input
        masked_image_latents = masked_image_latents.to(
            device=device, dtype=dtype)
        return mask, masked_image_latents

    # Copied from diffusers.pipelines.stable_diffusion.
    # pipeline_stable_diffusion_img2img.
    # StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(
            int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def __call__(
        self,
        promptA: Union[str, List[str]] = None,
        promptB: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        tradoff: float = 1.0,
        tradoff_nag: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_promptA: Optional[Union[str, List[str]]] = None,
        negative_promptB: Optional[Union[str, List[str]]] = None,
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
        task_class: Union[torch.Tensor, float, int] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If
                not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to be
                inpainted (which parts of the image to be masked
                out with `mask_image` and repainted according to `prompt`).
            mask_image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to mask
                `image`. White pixels in the mask are repainted
                while black pixels are preserved. If `mask_image` is a
                PIL image, it is converted to a single channel
                (luminance) before use. If it's a tensor, it should contain
                one color channel (L) instead of 3, so the
                expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to `self.unet.config.
                sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.
                sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be
                between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the
                `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1,
                added noise is maximum and the denoising
                process runs for the full number of iterations specified in
                `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually
                lead to a higher quality image at the
                expense of slower inference. This parameter is modulated
                by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to
                generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale
                is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in
                image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not
                using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM]
                (https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is
                ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`,
                *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable
                /generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian
                distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation
                with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied
                random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily
                tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the
                `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to
                easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated
                from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between
                `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.
                StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during
                inference. The function is called with the
                following arguments: `callback(step: int, timestep: int,
                latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called.
                If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the
                [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers
                /blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionInpaintPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://raw.githubusercontent.com/CompVis/
            latent-diffusion/main/data/inpainting_examples
            /overture-creations-5sI6fQgYIuo.png"
        >>> mask_url = "https://raw.githubusercontent.com/CompVis/
            latent-diffusion/main/data/inpainting_examples/
            overture-creations-5sI6fQgYIuo_mask.png"

        >>> init_image = download_image(img_url).resize((512, 512))
        >>> mask_image = download_image(mask_url).resize((512, 512))

        >>> pipe = StableDiffusionInpaintPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "Face of a yellow cat, high resolution,
                sitting on a park bench"
        >>> image = pipe(prompt=prompt, image=init_image,
                mask_image=mask_image).images[0]
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]
                or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.
                StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is
                a list with the generated images and the
                second element is a list of `bool`s indicating whether the
                corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        prompt = promptA
        negative_prompt = negative_promptA
        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance
        # weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
        # `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get('scale', None)
            if cross_attention_kwargs is not None else None)
        prompt_embeds = self._encode_prompt(
            promptA,
            promptB,
            tradoff,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_promptA,
            negative_promptB,
            tradoff_nag,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps,
            strength=strength,
            device=device)
        # check that number of inference steps is not < 1 -
        # as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f'After adjusting the num_inference_steps by strength '
                f' parameter: {strength}, the number of pipeline'
                f'steps is {num_inference_steps} which is < 1 and not '
                'appropriate for this pipeline.')
        # at which timestep to set the initial noise (n.b. 50% if
        # strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size *
                                               num_images_per_prompt)
        # create a boolean to check if the strength is set to 1.
        # if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        mask, masked_image, init_image = prepare_mask_and_masked_image(
            image, mask_image, height, width, return_image=True)
        mask_condition = mask.clone()

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + \
                    num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    'Incorrect configuration settings! The '
                    f' config of `pipeline.unet`: {self.unet.config} expects'
                    f' {self.unet.config.in_channels} but received '
                    f' `num_channels_latents`: {num_channels_latents} +'
                    f' `num_channels_mask`: {num_channels_mask} + '
                    ' `num_channels_masked_image`:'
                    f' {num_channels_masked_image}'
                    f' = {num_channels_latents+num_channels_masked_image+ num_channels_mask}. Please verify the config of'  # noqa
                    ' `pipeline.unet` or your `mask_image` or `image` input.')
        elif num_channels_unet != 4:
            raise ValueError(
                f'The unet {self.unet.__class__} should have either 4 or 9 '
                f'input channels, not {self.unet.config.in_channels}.')

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just
        # be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Denoising loop
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents
                # in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                if num_channels_unet == 9:
                    latent_model_input = torch.cat(
                        [latent_model_input, mask, masked_image_latents],
                        dim=1)

                # predict the noise residual
                if task_class is not None:
                    noise_pred = self.unet(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                        task_class=task_class,
                    )[0]
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False)[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents[:1]
                    init_mask = mask[:1]

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise,
                            torch.tensor([noise_timestep]))

                    latents = (1 - init_mask
                               ) * init_latents_proper + init_mask * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and
                    (i + 1) % self.scheduler.order == 0):  # noqa
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == 'latent':
            condition_kwargs = {}
            if isinstance(self.vae, AsymmetricAutoencoderKL):
                init_image = init_image.to(
                    device=device, dtype=masked_image_latents.dtype)
                init_image_condition = init_image.clone()
                init_image = self._encode_vae_image(
                    init_image, generator=generator)
                mask_condition = mask_condition.to(
                    device=device, dtype=masked_image_latents.dtype)
                condition_kwargs = {
                    'image': init_image_condition,
                    'mask': mask_condition
                }
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                **condition_kwargs)[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept)
