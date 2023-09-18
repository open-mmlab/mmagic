# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange
from mmengine import print_log
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from safetensors import safe_open
from tqdm import tqdm

from mmagic.models.archs import TokenizerWrapper, set_lora
from mmagic.models.utils import build_module, set_tomesd, set_xformers
from mmagic.registry import DIFFUSION_SCHEDULERS, MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList
from .animatediff_utils import (convert_ldm_clip_checkpoint,
                                convert_ldm_unet_checkpoint,
                                convert_ldm_vae_checkpoint)

logger = MMLogger.get_current_instance()

ModelType = Union[Dict, nn.Module]


@MODELS.register_module('animatediff')
@MODELS.register_module()
class AnimateDiff(BaseModel):
    """Implementation of `AnimateDiff.

    <https://arxiv.org/abs/2307.04725>`_ (AnimateDiff).

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
        fine_tune_text_encoder (bool, optional): Whether to fine-tune text
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

    def __init__(
        self,
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
        motion_module_cfg: Optional[dict] = None,
        dream_booth_lora_cfg: Optional[dict] = None,
    ):
        super().__init__(data_preprocessor)

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

        self.init_motion_module(motion_module_cfg)

        self.scheduler = build_module(scheduler, DIFFUSION_SCHEDULERS)
        if test_scheduler is None:
            self.test_scheduler = deepcopy(self.scheduler)
        else:
            self.test_scheduler = build_module(test_scheduler,
                                               DIFFUSION_SCHEDULERS)
        self.text_encoder = build_module(text_encoder, MODELS)
        if not isinstance(tokenizer, str):
            self.tokenizer = tokenizer
        else:
            # NOTE: here we assume tokenizer is an string
            self.tokenizer = TokenizerWrapper(tokenizer, subfolder='tokenizer')

        self.unet_sample_size = self.unet.sample_size
        self.vae_scale_factor = 2**(len(self.vae.block_out_channels) - 1)

        self.enable_noise_offset = noise_offset_weight > 0
        self.noise_offset_weight = noise_offset_weight

        self.enable_xformers = enable_xformers
        self.unet.set_use_memory_efficient_attention_xformers(True)

        self.tomesd_cfg = tomesd_cfg
        self.set_tomesd()

        self.init_dreambooth_lora(dream_booth_lora_cfg)

        self.prepare_model()

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
        """Set device for the model."""
        return next(self.parameters()).device

    def init_motion_module(self, motion_module_cfg):
        if motion_module_cfg is not None:
            if 'path' in motion_module_cfg.keys():
                motion_module_state_dict = torch.load(
                    motion_module_cfg['path'], map_location='cpu')
                # if "global_step" in motion_module_state_dict:
                # func_args.update({"global_step":
                # motion_module_state_dict["global_step"]})
                missing, unexpected = self.unet.load_state_dict(
                    motion_module_state_dict, strict=False)
                assert len(unexpected) == 0

    def init_dreambooth_lora(self, dream_booth_lora_cfg):
        # TODO: finish
        if dream_booth_lora_cfg is not None:
            if 'path' in dream_booth_lora_cfg.keys():
                state_dict = {}
                with safe_open(
                        dream_booth_lora_cfg['path'], framework='pt',
                        device='cpu') as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                # vae
                converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                    state_dict, self.vae.config)
                self.vae.load_state_dict(converted_vae_checkpoint)
                # unet
                converted_unet_checkpoint = convert_ldm_unet_checkpoint(
                    state_dict, self.unet.config)
                self.unet.load_state_dict(
                    converted_unet_checkpoint, strict=False)
                # text_model
                self.text_encoder = convert_ldm_clip_checkpoint(state_dict)
                # self.convert_lora(state_dict)

    def _encode_prompt(self, prompt, device, num_videos_per_prompt,
                       do_classifier_free_guidance, negative_prompt):
        """Encodes the prompt into text encoder hidden states."""
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
            prompt, padding='longest', return_tensors='pt').input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
            logger.warning(
                'The following part of your input was truncated '
                f'because CLIP can only handle sequences up to'
                f' {self.tokenizer.model_max_length} tokens: {removed_text}')

        text_encoder = self.text_encoder.module if hasattr(
            self.text_encoder, 'module') else self.text_encoder
        if hasattr(text_encoder.config, 'use_attention_mask'
                   ) and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation
        # per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f'`negative_prompt` should be the same type '
                    f'to `prompt`, but got {type(negative_prompt)} !='
                    f' {type(prompt)}.')
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has '
                    f'batch size {len(negative_prompt)}, but `prompt`:'
                    f' {prompt} has batch size {batch_size}. Please '
                    f'make sure that passed `negative_prompt` matches'
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

            if hasattr(text_encoder.config, 'use_attention_mask'
                       ) and text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation
            # per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings
            # into a single batch to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        """latents decoder."""
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, 'b c f h w -> (b f) c h w')
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(
                self.vae.decode(latents[frame_idx:frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, '(b f) c h w -> b c f h w', f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant
        # overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        """Prepare extra kwargs for the scheduler step, since not all
        schedulers have the same signature eta (η) is only used with the
        DDIMScheduler, it will be ignored for other schedulers."""
        # prepare extra kwargs for the scheduler step, since not all
        # schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will
        # be ignored for other schedulers.
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

    def check_inputs(self, prompt, height, width):
        """Check inputs.

        Raise error if not correct
        """
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f'`prompt` has to be of type `str`'
                             f' or `list` but is {type(prompt)}')

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f'`height` and `width` have to be divisible'
                             f' by 8 but are {height} and {width}.')

        # if (callback_steps is None) or (
        #     callback_steps is not None and (not isinstance(callback_steps,
        #       int) or callback_steps <= 0)
        # ):
        #     raise ValueError(
        #         f"`callback_steps` has to be a positive integer but
        #         is {callback_steps} of type"
        #         f" {type(callback_steps)}."
        #     )

    def convert_lora(self,
                     state_dict,
                     LORA_PREFIX_UNET='lora_unet',
                     LORA_PREFIX_TEXT_ENCODER='lora_te',
                     alpha=0.6):
        """ Convert lora for unet and text_encoder
            TODO: use this function to convert lora

        Args:
            state_dict (_type_): _description_
            LORA_PREFIX_UNET (str, optional):
            _description_. Defaults to 'lora_unet'.
            LORA_PREFIX_TEXT_ENCODER (str, optional):
            _description_. Defaults to 'lora_te'.
            alpha (float, optional): _description_. Defaults to 0.6.

        Returns:
            TODO: check each output type
            _type_: unet && text_encoder
        """
        # load base model
        # pipeline = StableDiffusionPipeline.from_pretrained(base_model_path,
        # torch_dtype=torch.float32)

        # load LoRA weight from .safetensors
        # state_dict = load_file(checkpoint_path)

        visited = []

        # directly update weight in diffusers model
        for key in state_dict:
            # it is suggested to print out the key, it usually
            # will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            # as we have set the alpha beforehand, so just skip
            if '.alpha' in key or key in visited:
                continue

            if 'text' in key:
                layer_infos = key.split('.')[0].split(
                    LORA_PREFIX_TEXT_ENCODER + '_')[-1].split('_')
                curr_layer = self.text_encoder
            else:
                layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET +
                                                      '_')[-1].split('_')
                curr_layer = self.unet
            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += '_' + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            pair_keys = []
            if 'lora_down' in key:
                pair_keys.append(key.replace('lora_down', 'lora_up'))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace('lora_up', 'lora_down'))

            # update weight
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(
                    torch.float32)
                weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(
                    2).to(torch.float32)
                curr_layer.weight.data += alpha * torch.mm(
                    weight_up, weight_down).unsqueeze(2).unsqueeze(3).to(
                        curr_layer.weight.data.device)
            else:
                weight_up = state_dict[pair_keys[0]].to(torch.float32)
                weight_down = state_dict[pair_keys[1]].to(torch.float32)
                curr_layer.weight.data += alpha * torch.mm(
                    weight_up, weight_down).to(curr_layer.weight.data.device)

            # update visited list
            for item in pair_keys:
                visited.append(item)

        return self.unet, self.text_encoder

    def prepare_latents(self,
                        batch_size,
                        num_channels_latents,
                        video_length,
                        height,
                        width,
                        dtype,
                        device,
                        generator,
                        latents=None):
        """Prepare latent variables."""
        shape = (batch_size, num_channels_latents, video_length,
                 height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length '
                f'{len(generator)}, but requested an effective batch'
                f' size of {batch_size}. Make sure the batch size matches the'
                f' length of the generators.')
        if latents is None:
            rand_device = 'cpu' if device.type == 'mps' else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(
                        shape,
                        generator=generator[i],
                        device=rand_device,
                        dtype=dtype) for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(
                    shape,
                    generator=generator,
                    device=rand_device,
                    dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f'Unexpected latents shape, got '
                                 f'{latents.shape}, expected {shape}')
            latents = latents.to(device)

        # scale the initial noise by the standard deviation
        # required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_model(self):
        """Prepare model for training.

        Move model to target dtype and disable gradient for some models.
        """
        self.vae.requires_grad_(False)
        print_log('Set VAE untrainable.', 'current')
        self.vae.to(self.dtype)
        print_log(f'Move VAE to {self.dtype}.', 'current')
        # if not self.finetune_text_encoder or self.lora_config:
        if 1:
            self.text_encoder.requires_grad_(False)
            print_log('Set Text Encoder untrainable.', 'current')
            self.text_encoder.to(self.dtype)
            print_log(f'Move Text Encoder to {self.dtype}.', 'current')
        # if self.lora_config:
        if 1:
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

    @torch.no_grad()
    def infer(self,
              prompt: Union[str, List[str]],
              video_length: Optional[int] = 16,
              height: Optional[int] = None,
              width: Optional[int] = None,
              num_inference_steps: int = 50,
              guidance_scale: float = 7.5,
              negative_prompt: Optional[Union[str, List[str]]] = None,
              num_videos_per_prompt: Optional[int] = 1,
              eta: float = 0.0,
              generator: Optional[Union[torch.Generator,
                                        List[torch.Generator]]] = None,
              latents: Optional[torch.FloatTensor] = None,
              return_type: Optional[str] = 'tensor',
              show_progress: bool = True,
              seed: Optional[int] = 1007):
        """Function invoked when calling the pipeline for generation.

        Args:
            prompt (str or List[str]): The prompt or prompts to guide
                the video generation.
            video_length (int, Option): The number of frames of the
                generated video. Defaults to 16.
            height (int, Optional): The height in pixels of the generated
                image. If not passed, the height will be
                `self.unet_sample_size * self.vae_scale_factor` Defaults
                to None.
            width (int, Optional): The width in pixels of the generated image.
                If not passed, the width will be
                `self.unet_sample_size * self.vae_scale_factor` Defaults
                to None.
            num_inference_steps (int): The number of denoising steps.
                More denoising steps usually lead to a higher quality video at
                the expense of slower inference. Defaults to 50.
            guidance_scale (float): Guidance scale as defined in Classifier-
                Free Diffusion Guidance (https://arxiv.org/abs/2207.12598).
                Defaults to 7.5
            negative_prompt (str or List[str], optional): The prompt or
                prompts not to guide the video generation. Ignored when not
                using guidance (i.e., ignored if `guidance_scale` is less
                than 1). Defaults to None.
            num_videos_per_prompt (int): The number of videos to generate
                per prompt. Defaults to 1.
            eta (float): Corresponds to parameter eta (η) in the DDIM paper:
                https://arxiv.org/abs/2010.02502. Only applies to
                DDIMScheduler, will be ignored for others. Defaults to 0.0.
            generator (torch.Generator, optional): A torch generator to make
                generation deterministic. Defaults to None.
            latents (torch.FloatTensor, optional): Pre-generated noisy latents,
                sampled from a Gaussian distribution, to be used as inputs for
                video generation. Can be used to tweak the same generation with
                different prompts. If not provided, a latents tensor will be
                generated by sampling using the supplied random `generator`.
                Defaults to None.
            return_type (str): The return type of the inference results.
                Supported types are 'video', 'numpy', 'tensor'. If 'video'
                is passed, a list of PIL images will be returned. If 'numpy'
                is passed, a numpy array with shape [N, C, H, W] will be
                returned, and the value range will be same as decoder's
                output range. If 'tensor' is passed, the decoder's output
                will be returned. Defaults to 'image'.
        #TODO
        Returns:
            dict: A dict containing the generated video
        """
        assert return_type in ['image', 'tensor', 'numpy']

        # 0. Default height and width to unet
        height = height or self.unet_sample_size * self.vae_scale_factor
        width = width or self.unet_sample_size * self.vae_scale_factor
        if seed != -1:
            torch.manual_seed(seed)
        print_log(f'current seed: {torch.initial_seed()}')
        print_log(f'sampling {prompt} ...')

        # 1. Check inputs. Raise error if not correct

        self.check_inputs(prompt, height,
                          width)  # NOTE: aligned with origin repo

        # 2. Define call parameters
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)
        device = self.device

        # here `guidance_scale` is defined analog to the
        # guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
        # `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        video_dtype = self.vae.module.dtype if hasattr(self.vae, 'module') \
            else self.vae.dtype

        # 3. Encode input prompt

        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(
                negative_prompt, list) else [negative_prompt] * batch_size
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance,
            negative_prompt)  # NOTE aligned with origin repo

        # 4. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.test_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.test_scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )  # NOTE aligned with origin repo
        latents_dtype = latents.dtype

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

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            )['sample'].to(dtype=latents_dtype)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.test_scheduler.step(
                    noise_pred, t, latents,
                    **extra_step_kwargs)['prev_sample']  # FIXME: not aligned
                # FIXME: revise config thresholding=False
                # scheduler pred_original_sample not aligned
                # fixed clip_sample=False
        # 8. Post-processing
        video = self.decode_latents(latents.to(video_dtype))

        if return_type == 'tensor':
            video = torch.from_numpy(video)

        return {'samples': video}

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """forward is not implemented now."""
        raise NotImplementedError(
            'Forward is not implemented now, please use infer.')
