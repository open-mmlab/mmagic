# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapperDict
from mmengine.runner import set_random_seed
from PIL import Image
from tqdm.auto import tqdm

from mmagic.models.archs import TokenizerWrapper, set_lora
from mmagic.models.utils import build_module, set_tomesd, set_xformers
from mmagic.registry import DIFFUSION_SCHEDULERS, MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList

logger = MMLogger.get_current_instance()

ModelType = Union[Dict, nn.Module]


@MODELS.register_module('sdxl')
@MODELS.register_module()
class StableDiffusionXL(BaseModel):
    """Class for Stable Diffusion XL. Refers to https://github.com/Stability-
    AI.

    /generative-models and https://github.com/huggingface/diffusers/blob/main/
    src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py

    Args:
        unet (Union[dict, nn.Module]): The config or module for Unet model.
        text_encoder_one (Union[dict, nn.Module]): The config or module for
            text encoder.
        tokenizer_one (str): The **name** for CLIP tokenizer.
        text_encoder_two (Union[dict, nn.Module]): The config or module for
            text encoder.
        tokenizer_two (str): The **name** for CLIP tokenizer.
        vae (Union[dict, nn.Module]): The config or module for VAE model.
        schedule (Union[dict, nn.Module]): The config or module for diffusion
            scheduler.
        test_scheduler (Union[dict, nn.Module], optional): The config or
            module for diffusion scheduler in test stage (`self.infer`). If not
            passed, will use the same scheduler as `schedule`. Defaults to
            None.
        dtype (str, optional): The dtype for the model This argument will not
            work when dtype is defined for submodels. Defaults to None.
        enable_xformers (bool, optional): Whether to use xformers.
            Defaults to True.
        noise_offset_weight (bool, optional): The weight of noise offset
            introduced in
            https://www.crosslabs.org/blog/diffusion-with-offset-noise
            Defaults to 0.
        tomesd_cfg (dict, optional): The config for TOMESD. Please refers to
            https://github.com/dbolya/tomesd and
            https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/utils/tome_utils.py for detail.  # noqa
            Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        lora_config (dict, optional): The config for LoRA finetuning. Defaults
            to None.
        val_prompts (Union[str, List[str]], optional): The prompts for
            validation. Defaults to None.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. Defaults to False.
        force_zeros_for_empty_prompt (bool): Whether the negative prompt
            embeddings shall be forced to always be set to 0.
            Defaults to True.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """

    def __init__(self,
                 vae: ModelType,
                 text_encoder_one: ModelType,
                 tokenizer_one: str,
                 text_encoder_two: ModelType,
                 tokenizer_two: str,
                 unet: ModelType,
                 scheduler: ModelType,
                 test_scheduler: Optional[ModelType] = None,
                 dtype: Optional[str] = None,
                 enable_xformers: bool = True,
                 noise_offset_weight: float = 0,
                 tomesd_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[ModelType] = dict(
                     type='DataPreprocessor'),
                 lora_config: Optional[dict] = None,
                 val_prompts: Union[str, List[str]] = None,
                 finetune_text_encoder: bool = False,
                 force_zeros_for_empty_prompt: bool = True,
                 init_cfg: Optional[dict] = None):

        # TODO: support `from_pretrained` for this class
        super().__init__(data_preprocessor, init_cfg)

        default_args = dict()
        if dtype is not None:
            default_args['dtype'] = dtype

        self.dtype = torch.float32
        if dtype in ['float16', 'fp16', 'half']:
            self.dtype = torch.float16
        elif dtype == 'bf16':
            self.dtype = torch.bfloat16
        else:
            assert dtype in [
                'fp32', None
            ], ('dtype must be one of \'fp32\', \'fp16\', \'bf16\' or None.')

        self.vae = build_module(vae, MODELS, default_args=default_args)
        self.unet = build_module(unet, MODELS)  # NOTE: initialize unet as fp32
        self._unet_ori_dtype = next(self.unet.parameters()).dtype
        print_log(f'Set UNet dtype to \'{self._unet_ori_dtype}\'.', 'current')
        self.scheduler = build_module(scheduler, DIFFUSION_SCHEDULERS)
        if test_scheduler is None:
            self.test_scheduler = deepcopy(self.scheduler)
        else:
            self.test_scheduler = build_module(test_scheduler,
                                               DIFFUSION_SCHEDULERS)
        self.text_encoder_one = build_module(text_encoder_one, MODELS)
        if not isinstance(tokenizer_one, str):
            self.tokenizer_one = tokenizer_one
        else:
            # NOTE: here we assume tokenizer is an string
            self.tokenizer_one = TokenizerWrapper(
                tokenizer_one, subfolder='tokenizer')

        self.text_encoder_two = build_module(text_encoder_two, MODELS)
        if not isinstance(tokenizer_two, str):
            self.tokenizer_two = tokenizer_two
        else:
            # NOTE: here we assume tokenizer is an string
            self.tokenizer_two = TokenizerWrapper(
                tokenizer_two, subfolder='tokenizer_2')

        self.unet_sample_size = self.unet.sample_size
        self.vae_scale_factor = 2**(len(self.vae.block_out_channels) - 1)

        self.enable_noise_offset = noise_offset_weight > 0
        self.noise_offset_weight = noise_offset_weight

        self.finetune_text_encoder = finetune_text_encoder
        self.val_prompts = val_prompts
        self.lora_config = deepcopy(lora_config)
        self.force_zeros_for_empty_prompt = force_zeros_for_empty_prompt

        self.prepare_model()
        self.set_lora()

        self.enable_xformers = enable_xformers
        self.set_xformers()

        self.tomesd_cfg = tomesd_cfg
        self.set_tomesd()

    def prepare_model(self):
        """Prepare model for training.

        Move model to target dtype and disable gradient for some models.
        """
        self.vae.requires_grad_(False)
        print_log('Set VAE untrainable.', 'current')
        self.vae.to(self.dtype)
        print_log(f'Move VAE to {self.dtype}.', 'current')
        if not self.finetune_text_encoder or self.lora_config:
            self.text_encoder_one.requires_grad_(False)
            self.text_encoder_two.requires_grad_(False)
            print_log('Set Text Encoder untrainable.', 'current')
            self.text_encoder_one.to(self.dtype)
            self.text_encoder_two.to(self.dtype)
            print_log(f'Move Text Encoder to {self.dtype}.', 'current')
        if self.lora_config:
            self.unet.requires_grad_(False)
            print_log('Set Unet untrainable.', 'current')

    def set_lora(self):
        """Set LORA for model."""
        if self.lora_config:
            set_lora(self.unet, self.lora_config)

    def set_xformers(self, module: Optional[nn.Module] = None) -> nn.Module:
        """Set xformers for the model.

        Returns:
            nn.Module: The model with xformers.
        """
        if self.enable_xformers:
            if module is None:
                set_xformers(self)
            else:
                set_xformers(module)

    def set_tomesd(self) -> nn.Module:
        """Set ToMe for the stable diffusion model.

        Returns:
            nn.Module: The model with ToMe.
        """
        if self.tomesd_cfg is not None:
            set_tomesd(self, **self.tomesd_cfg)

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode: bool = True):
        """Set train/eval mode.

        Args:
            mode (bool, optional): Whether set train mode. Defaults to True.
        """
        if mode:
            if next(self.unet.parameters()).dtype != self._unet_ori_dtype:
                print_log(
                    f'Set UNet dtype to \'{self._unet_ori_dtype}\' '
                    'in the train mode.', 'current')
            self.unet.to(self._unet_ori_dtype)
        else:
            self.unet.to(self.dtype)
            print_log(f'Set UNet dtype to \'{self.dtype}\' in the eval mode.',
                      'current')
        return super().train(mode)

    @torch.no_grad()
    def infer(self,
              prompt: Union[str, List[str]],
              prompt_2: Optional[Union[str, List[str]]] = None,
              height: Optional[int] = None,
              width: Optional[int] = None,
              num_inference_steps: int = 50,
              denoising_end: Optional[float] = None,
              guidance_scale: float = 7.5,
              negative_prompt: Optional[Union[str, List[str]]] = None,
              negative_prompt_2: Optional[Union[str, List[str]]] = None,
              num_images_per_prompt: Optional[int] = 1,
              eta: float = 0.0,
              generator: Optional[torch.Generator] = None,
              latents: Optional[torch.FloatTensor] = None,
              show_progress: bool = True,
              seed: int = 1,
              original_size: Optional[Tuple[int, int]] = None,
              crops_coords_top_left: Tuple[int, int] = (0, 0),
              target_size: Optional[Tuple[int, int]] = None,
              negative_original_size: Optional[Tuple[int, int]] = None,
              negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
              negative_target_size: Optional[Tuple[int, int]] = None,
              return_type='image'):
        """Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            prompt2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_two` and
                `text_encoder_two`. If not defined, `prompt` is used in both
                text-encoders. Defaults to None.
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
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0)
                of the total denoising process to be completed before it is
                intentionally prematurely terminated. As a result, the returned
                sample will still retain a substantial amount of noise as
                determined by the discrete timesteps selected by the scheduler.
                The denoising_end parameter should ideally be utilized when
                this pipeline forms a part of a "Mixture of Denoisers"
                multi-pipeline setup, as elaborated in
                [**Refining the Image Output**](
                https://huggingface.co/docs/diffusers/api/pipelines/
                stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in
                [Classifier-Free Diffusion Guidance]
                (https://arxiv.org/abs/2207.12598).
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
                Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*)):
                The `negative_prompt` to be sent to the `tokenizer_two` and
                `text_encoder_two`. If not defined, `negative_prompt` is used
                in both text-encoders. Defaults to None.
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
            show_progress (bool): Whether to show progress.
                Defaults to False.
            seed (int): Seed to be used. Defaults to 1.
            original_size (`Tuple[int]`, *optional*):
                If `original_size` is not the same as `target_size` the image
                will appear to be down- or upsampled. If `original_size` is
                `(width, height)` if not specified.
                Defaults to None.
            crops_coords_top_left (`Tuple[int]`, *optional*):
                `crops_coords_top_left` can be used to generate an image that
                appears to be "cropped" from the position. Favorable,
                well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0).
                Defaults to (0, 0).
            target_size (`Tuple[int]`, *optional*):
                For most cases, `target_size` should be set to the desired
                height and width of the generated image. If not specified it
                will be `(width, height)`. Defaults to None.
            negative_original_size (`Tuple[int]`, *optional*):
                To negatively condition the generation process based on a
                specific image resolution. For more information, refer to this
                issue thread:
                https://github.com/huggingface/diffusers/issues/4208.
                Defaults to None.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*):
                To negatively condition the generation process based on a
                specific crop coordinates. For more information, refer to this
                issue thread:
                https://github.com/huggingface/diffusers/issues/4208.
                Defaults to (0, 0).
            negative_target_size (`Tuple[int]`, *optional*):
                To negatively condition the generation process based on a
                target image resolution. It should be as same as the
                `target_size` for most cases. For more information,
                refer to this issue thread:
                https://github.com/huggingface/diffusers/issues/4208.
                Defaults to None.
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

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self.device

        img_dtype = self.vae.module.dtype if hasattr(self.vae, 'module') \
            else self.vae.dtype
        latent_dtype = next(self.unet.parameters()).dtype
        # here `guidance_scale` is defined analog to the
        # guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
        # `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt(prompt, prompt_2, device,
                                num_images_per_prompt,
                                do_classifier_free_guidance, negative_prompt,
                                negative_prompt_2)

        # 4. Prepare timesteps
        self.test_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.test_scheduler.timesteps

        # 5. Prepare latent variables
        if hasattr(self.unet, 'module'):
            num_channels_latents = self.unet.module.in_channels
        else:
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

        # 6. Prepare extra step kwargs.
        # TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype)
        if (negative_original_size is not None) and (negative_target_size
                                                     is not None):
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds],
                                      dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids],
                                     dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1)

        # 9 Apply denoising_end
        if denoising_end is not None and isinstance(
                denoising_end,
                float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(self.scheduler.config.num_train_timesteps -
                      (denoising_end *
                       self.scheduler.config.num_train_timesteps)))
            num_inference_steps = len(
                list(
                    filter(lambda ts: ts >= discrete_timestep_cutoff,
                           timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 10. Denoising loop
        if show_progress:
            timesteps = tqdm(timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.test_scheduler.scale_model_input(
                latent_model_input, t)

            latent_model_input = latent_model_input.to(latent_dtype)
            prompt_embeds = prompt_embeds.to(latent_dtype)
            # predict the noise residual
            added_cond_kwargs = {
                'text_embeds': add_text_embeds,
                'time_ids': add_time_ids
            }
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

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
        if return_type == 'image':
            image = self.output_to_pil(image)
        elif return_type == 'numpy':
            image = image.cpu().numpy()
        else:
            assert return_type == 'tensor', (
                'Only support \'image\', \'numpy\' and \'tensor\' for '
                f'return_type, but receive {return_type}')

        return {'samples': image}

    def _get_add_time_ids(self, original_size: Optional[Tuple[int, int]],
                          crops_coords_top_left: Tuple[int, int],
                          target_size: Optional[Tuple[int, int]], dtype):
        """Get `add_time_ids`.

        Args:
            original_size (`Tuple[int]`, *optional*):
                If `original_size` is not the same as `target_size` the image
                will appear to be down- or upsampled. If `original_size` is
                `(width, height)` if not specified.
                Defaults to None.
            crops_coords_top_left (`Tuple[int]`, *optional*):
                `crops_coords_top_left` can be used to generate an image that
                appears to be "cropped" from the position. Favorable,
                well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0).
                Defaults to (0, 0).
            target_size (`Tuple[int]`, *optional*):
                For most cases, `target_size` should be set to the desired
                height and width of the generated image. If not specified it
                will be `(width, height)`. Defaults to None.
            dtype (str, optional): The dtype for the embeddings.

        Returns:
            add_time_ids (torch.Tensor): time ids for time embeddings layer.
        """
        add_time_ids = list(original_size + crops_coords_top_left +
                            target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) +
            self.text_encoder_two.config.projection_dim)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                'Model expects an added time embedding vector of length '
                f'{expected_add_embed_dim}, but a vector of '
                f'{passed_add_embed_dim} was created. The model has an '
                'incorrect config. Please check '
                '`unet.config.time_embedding_type` and '
                '`text_encoder_2.config.projection_dim`.')

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def output_to_pil(self, image) -> List[Image.Image]:
        """Convert output tensor to PIL image. Output tensor will be de-normed
        to [0, 255] by `DataPreprocessor.destruct`. Due to no `data_samples` is
        passed, color order conversion will not be performed.

        Args:
            image (torch.Tensor): The output tensor of the decoder.

        Returns:
            List[Image.Image]: The list of processed PIL images.
        """
        image = self.data_preprocessor.destruct(image)
        image = image.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        image = [Image.fromarray(img) for img in image]
        return image

    def _encode_prompt(self, prompt, prompt_2, device, num_images_per_prompt,
                       do_classifier_free_guidance, negative_prompt,
                       negative_prompt_2):
        """Encodes the prompt into text encoder hidden states.

        Args:
            prompt (str or list(int)): prompt to be encoded.
            prompt_2 (str or list(int)): prompt to be encoded. Send to the
                `tokenizer_two` and `text_encoder_two`. If not defined,
                `prompt` is used in both text-encoders.
            device: (torch.device): torch device.
            num_images_per_prompt (int): number of images that should be
                generated per prompt.
            do_classifier_free_guidance (`bool`): whether to use classifier
                free guidance or not.
            negative_prompt (str or List[str]): The prompt or prompts not
                to guide the image generation. Ignored when not using
                guidance (i.e., ignored if `guidance_scale` is less than `1`).
            negative_prompt_2 (str or List[str]): The prompt or prompts not
                to guide the image generation. Ignored when not using
                guidance (i.e., ignored if `guidance_scale` is less than `1`).
                Send to `tokenizer_two` and `text_encoder_two`. If not defined,
                `negative_prompt` is used in both text-encoders

        Returns:
            text_embeddings (torch.Tensor): text embeddings generated by
                clip text encoder.
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        prompt_2 = prompt_2 or prompt
        tokenizers = [self.tokenizer_one, self.tokenizer_two]
        text_encoders = [self.text_encoder_one, self.text_encoder_two]
        prompts = [prompt, prompt_2]
        prompt_embeds_list = []
        for tokenizer, text_encoder, prompt in zip(tokenizers, text_encoders,
                                                   prompts):
            text_inputs = tokenizer(
                prompt,
                padding='max_length',
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(
                prompt, padding='max_length', return_tensors='pt').input_ids

            if not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1:-1])
                logger.warning(
                    'The following part of your input was truncated because '
                    ' CLIP can only handle sequences up to'
                    f' {tokenizer.model_max_length} tokens: {removed_text}')

            text_encoder = text_encoder.module if hasattr(
                text_encoder, 'module') else text_encoder
            text_embeddings = text_encoder(
                text_input_ids.to(device),
                output_hidden_states=True,
            )
            pooled_prompt_embeds = text_embeddings.pooler_output
            text_embeddings = text_embeddings.hidden_states[-2]

            prompt_embeds_list.append(text_embeddings)

        text_embeddings = torch.concat(prompt_embeds_list, dim=-1)

        # duplicate text embeddings for each generation per prompt,
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and self.force_zeros_for_empty_prompt:
            negative_prompt_embeds = torch.zeros_like(text_embeddings)
            negative_pooled_prompt_embeds = torch.zeros_like(
                pooled_prompt_embeds)
        elif do_classifier_free_guidance:
            negative_prompt = negative_prompt or ''
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    '`negative_prompt` should be the same type to `prompt`, '
                    f'but got {type(negative_prompt)} != {type(prompt)}.')
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt, negative_prompt_2]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has batch size '
                    f'{len(negative_prompt)}, but `prompt`: {prompt} has batch'
                    f' size {batch_size}. Please make sure that passed '
                    '`negative_prompt` matches the batch size of `prompt`.')
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(
                    uncond_tokens, tokenizers, text_encoders):
                max_length = text_embeddings.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding='max_length',
                    max_length=max_length,
                    truncation=True,
                    return_tensors='pt',
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the
                # final text encoder
                negative_pooled_prompt_embeds = (
                    negative_prompt_embeds.pooler_output)
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[
                    -2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(
                negative_prompt_embeds_list, dim=-1)

        bs_embed, seq_len, _ = text_embeddings.shape
        # duplicate text embeddings for each generation per prompt, using mps
        # friendly method
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt
            # ,using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            1, num_images_per_prompt).view(bs_embed * num_images_per_prompt,
                                           -1)
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = (
                negative_pooled_prompt_embeds.repeat(
                    1, num_images_per_prompt).view(
                        bs_embed * num_images_per_prompt, -1))

        return (text_embeddings, negative_prompt_embeds, pooled_prompt_embeds,
                negative_pooled_prompt_embeds)

    def decode_latents(self, latents):
        """use vae to decode latents.

        Args:
            latents (torch.Tensor): latents to decode.

        Returns:
            image (torch.Tensor): image result.
        """
        latents = 1 / 0.18215 * latents
        if hasattr(self.vae, 'module'):
            image = self.vae.module.decode(latents)['sample']
        else:
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

    def prepare_test_scheduler_extra_step_kwargs(self, generator, eta):
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
            inspect.signature(self.test_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.test_scheduler.step).parameters.keys())
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

    @torch.no_grad()
    def val_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More details in `Metrics` and `Evaluator`.

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
        if self.val_prompts is None:
            gt_img = self.data_preprocessor.destruct(data['inputs'],
                                                     data_samples)

            out_data_sample = DataSample(
                fake_img=samples, gt_img=gt_img, prompt=prompt)
        else:
            out_data_sample = DataSample(fake_img=samples, prompt=prompt)

        data_sample_list = out_data_sample.split()
        return data_sample_list

    @torch.no_grad()
    def test_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More details in `Metrics` and `Evaluator`.

        Returns:
            SampleList: A list of ``DataSample`` contain generated results.
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
        if self.val_prompts is None:
            gt_img = self.data_preprocessor.destruct(data['inputs'],
                                                     data_samples)

            out_data_sample = DataSample(
                fake_img=samples, gt_img=gt_img, prompt=prompt)
        else:
            out_data_sample = DataSample(fake_img=samples, prompt=prompt)

        data_sample_list = out_data_sample.split()
        return data_sample_list

    def encode_prompt_train(self, text_one, text_two):
        """Encode prompt for training.

        Args:
            text_one (torch.tensor): Input ids from tokenizer_one.
            text_two (torch.tensor): Input ids from tokenizer_two.

        Returns:
            prompt_embeds (torch.tensor): Prompt embedings.
            pooled_prompt_embeds (torch.tensor): Pooled prompt embeddings.
        """
        prompt_embeds_list = []

        text_encoders = [self.text_encoder_one, self.text_encoder_two]
        texts = [text_one, text_two]
        for text_encoder, text in zip(text_encoders, texts):

            prompt_embeds = text_encoder(
                text,
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the
            # final text encoder
            pooled_prompt_embeds = prompt_embeds.pooler_output
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def train_step(self, data: List[dict], optim_wrapper: OptimWrapperDict):
        """Train step function.

        Args:
            data (List[dict]): Batch of data as input.
            optim_wrapper (OptimWrapperDict): Dict with optimizers
                for generator and discriminator (if have).

        Returns:
            dict: Dict with loss, information for logger, the number of \
                samples and results for visualization.
        """
        data = self.data_preprocessor(data)
        inputs, data_samples = data['inputs'], data['data_samples']

        vae = self.vae.module if hasattr(self.vae, 'module') else self.vae

        with optim_wrapper.optim_context(self.unet):
            image = inputs
            prompt = data_samples.prompt
            num_batches = image.shape[0]

            image = image.to(self.dtype)
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)

            if self.enable_noise_offset:
                noise = noise + self.noise_offset_weight * torch.randn(
                    latents.shape[0],
                    latents.shape[1],
                    1,
                    1,
                    device=noise.device)

            timesteps = torch.randint(
                0,
                self.scheduler.num_train_timesteps, (num_batches, ),
                device=self.device)
            timesteps = timesteps.long()

            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            input_ids_one = self.tokenizer_one(
                prompt,
                max_length=self.tokenizer_one.model_max_length,
                return_tensors='pt',
                padding='max_length',
                truncation=True)['input_ids'].to(self.device)
            input_ids_two = self.tokenizer_two(
                prompt,
                max_length=self.tokenizer_two.model_max_length,
                return_tensors='pt',
                padding='max_length',
                truncation=True)['input_ids'].to(self.device)

            (encoder_hidden_states,
             pooled_prompt_embeds) = self.encode_prompt_train(
                 input_ids_one, input_ids_two)
            unet_added_conditions = {
                'time_ids': data['time_ids'],
                'text_embeds': pooled_prompt_embeds
            }

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
                encoder_hidden_states=encoder_hidden_states.float(),
                added_cond_kwargs=unet_added_conditions)
            model_pred = model_output['sample']

            loss_dict = dict()
            # calculate loss in FP32
            loss_mse = F.mse_loss(model_pred.float(), gt.float())
            loss_dict['loss_mse'] = loss_mse

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
