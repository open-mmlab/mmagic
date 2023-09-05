# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner import set_random_seed
from PIL import Image
from tqdm.auto import tqdm

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList
from ..stable_diffusion.stable_diffusion import StableDiffusion
from .vico_utils import set_vico_modules

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class ViCo(StableDiffusion):
    """Implementation of `ViCo with Stable Diffusion.

    <https://arxiv.org/abs/2306.00971>`_ (ViCo).

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
        val_prompts (Union[str, List[str]], optional): The prompts for
            validation. Defaults to None.
        num_class_images (int, optional): The number of images for class prior.
            Defaults to 3.
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
        image_cross_layers (List[int], optional): The layers to use image
            cross attention. Defaults to None.
        reg_loss_weight (float, optional): The weight of regularization loss.
            Defaults to 0.
        placeholder (str, optional): The placeholder token. Defaults to None.
        initialize_token (str, optional): The token to initialize the
            placeholder. Defaults to None.
        num_vectors_per_token (int, optional): The number of vectors per token.
    """

    def __init__(self,
                 vae: ModelType,
                 text_encoder: ModelType,
                 tokenizer: str,
                 unet: ModelType,
                 scheduler: ModelType,
                 test_scheduler: Optional[ModelType] = None,
                 val_prompts: Union[str, List[str]] = None,
                 dtype: str = 'fp16',
                 enable_xformers: bool = True,
                 noise_offset_weight: float = 0,
                 tomesd_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[ModelType] = dict(
                     type='DataPreprocessor'),
                 init_cfg: Optional[dict] = None,
                 image_cross_layers: List[int] = None,
                 reg_loss_weight: float = 0,
                 placeholder: str = None,
                 initialize_token: str = None,
                 num_vectors_per_token: int = 1):

        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         test_scheduler, dtype, enable_xformers,
                         noise_offset_weight, tomesd_cfg, data_preprocessor,
                         init_cfg)
        self.reg_loss_weight = reg_loss_weight
        self.placeholder = placeholder

        self.dtype = torch.float32
        if dtype == 'fp16':
            self.dtype = torch.float16
        elif dtype == 'bf16':
            self.dtype = torch.bfloat16
        else:
            assert dtype in [
                'fp32', None
            ], ('dtype must be one of \'fp32\', \'fp16\', \'bf16\' or None.')

        self.val_prompts = val_prompts
        self.add_tokens(placeholder, initialize_token, num_vectors_per_token)
        self.set_vico(image_cross_layers)
        self.prepare_models()

    def prepare_models(self):
        """Prepare model for training.

        Move model to target dtype and disable gradient for some models.
        """
        """Disable gradient for untrainable modules to save memory."""
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.set_only_embedding_trainable()
        self.set_only_imca_trainable()

    def set_vico(self, have_image_cross_attention: List[int]):
        """Set ViCo for model."""
        set_vico_modules(self.unet, have_image_cross_attention)

    def set_only_imca_trainable(self):
        """Set only image cross attention trainable."""
        for _, layer in self.unet.named_modules():
            if layer.__class__.__name__ == 'ViCoTransformer2D':
                if hasattr(layer, 'image_cross_attention'):
                    layer.image_cross_attention.train()
                    for name, param in (
                            layer.image_cross_attention.named_parameters()):
                        param.requires_grad = True

    def add_tokens(self,
                   placeholder_token: str,
                   initialize_token: str = None,
                   num_vectors_per_token: int = 1):
        """Add token for training.

        # TODO: support add tokens as dict, then we can load pretrained tokens.
        """
        self.tokenizer.add_placeholder_token(
            placeholder_token, num_vec_per_token=num_vectors_per_token)

        self.text_encoder.set_embedding_layer()
        embedding_layer = self.text_encoder.get_embedding_layer()
        assert embedding_layer is not None, (
            'Do not support get embedding layer for current text encoder. '
            'Please check your configuration.')

        if initialize_token:
            init_id = self.tokenizer(initialize_token).input_ids[1]
            initialize_embedding = embedding_layer.weight[init_id]
            initialize_embedding = initialize_embedding[None, ...].repeat(
                num_vectors_per_token, 1)
        else:
            emb_dim = embedding_layer.weight.shape[1]
            initialize_embedding = torch.zeros(num_vectors_per_token, emb_dim)

        token_info = self.tokenizer.get_token_info(placeholder_token)
        token_info['embedding'] = initialize_embedding
        token_info['trainable'] = True
        self.token_info = token_info
        embedding_layer.add_embeddings(token_info)

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
        image_reference = data['inputs']['img_ref']
        data_samples = data['data_samples']
        if self.val_prompts is None:
            prompt = data_samples.prompt
        else:
            prompt = self.val_prompts
            # construct a fake data_sample for destruct
            data_samples.split() * len(prompt)
            data_samples = DataSample.stack(data_samples.split() * len(prompt))

        output = self.infer(prompt, image_reference, return_type='tensor')
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
        data = self.data_preprocessor(data)
        image_reference = data['inputs']['img_ref']
        data_samples = data['data_samples']

        if self.val_prompts is None:
            prompt = data_samples.prompt
        else:
            prompt = self.val_prompts
            # construct a fake data_sample for destruct
            data_samples.split() * len(prompt)
            data_samples = DataSample.stack(data_samples.split() * len(prompt))

        output = self.infer(prompt, image_reference, return_type='tensor')
        samples = output['samples']
        samples = self.data_preprocessor.destruct(samples, data_samples)

        out_data_sample = DataSample(fake_img=samples, prompt=prompt)
        data_sample_list = out_data_sample.split()
        return data_sample_list

    def prepare_reference(self,
                          image_ref: Union[Image.Image, torch.Tensor],
                          height: Optional[int] = 512,
                          width: Optional[int] = 512):
        if isinstance(image_ref, Image.Image):
            if not image_ref.mode == 'RGB':
                image_ref = image_ref.convert('RGB')
            img = np.array(image_ref).astype(np.uint8)
            image_ref = Image.fromarray(img)
            image_ref = image_ref.resize((height, width),
                                         resample=Image.BILINEAR)

            image_ref = np.array(image_ref).astype(np.uint8)
            image_ref = (image_ref / 127.5 - 1.0).astype(np.float32)
            image_ref = torch.from_numpy(image_ref).permute(2, 0,
                                                            1).unsqueeze(0)

        return image_ref

    def train_step(self, data, optim_wrapper):
        """Training step."""
        data = self.data_preprocessor(data)
        inputs, data_samples = data['inputs'], data['data_samples']

        with optim_wrapper.optim_context(self.unet):
            image = inputs['img']  # image for new concept
            num_batches = image.shape[0]

            image_ref = inputs['img_ref']
            # cat image and image reference to avoid forward twice
            image = torch.cat([image, image_ref], dim=0)
            prompt_init = data_samples.prompt
            placeholder_string = self.placeholder

            image = image.to(self.dtype)
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            noise = torch.randn_like(latents[:num_batches, ...])
            timesteps = torch.randint(
                0,
                self.scheduler.num_train_timesteps, (num_batches, ),
                device=self.device)
            timesteps = timesteps.long()
            # image reference shares the same timesteps

            # only add noise to source image
            noisy_latents = self.scheduler.add_noise(
                latents[:num_batches, ...], noise, timesteps)
            noisy_latents = torch.cat(
                [noisy_latents, latents[num_batches:, ...]], dim=0)
            timesteps = torch.cat([timesteps, timesteps])

            input_ids = self.tokenizer(
                prompt_init,
                max_length=self.tokenizer.model_max_length,
                return_tensors='pt',
                padding='max_length',
                truncation=True)['input_ids'].to(self.device)
            ph_tokens = self.tokenizer(
                [placeholder_string],
                max_length=self.tokenizer.model_max_length,
                return_tensors='pt',
                padding='max_length',
                truncation=True)['input_ids'].to(self.device)
            ph_tok = ph_tokens[0, 1]
            placeholder_idx = torch.where(input_ids == ph_tok)
            clip_eot_token_id = self.tokenizer.encode(
                self.tokenizer.eos_token)['input_ids'][1]
            endoftext_idx = (torch.arange(input_ids.shape[0]),
                             (input_ids == clip_eot_token_id).nonzero(
                                 as_tuple=False)[0, 1])
            placeholder_position = [placeholder_idx, endoftext_idx]

            encoder_hidden_states = self.text_encoder(input_ids)[0]
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, encoder_hidden_states], dim=0)
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
                placeholder_position=placeholder_position)
            model_pred = model_output['sample']
            loss_reg = model_output['loss_reg']

            loss_dict = dict()
            if loss_reg != 0:
                loss_dict['loss_reg'] = loss_reg * self.reg_loss_weight
            vico_loss = F.mse_loss(model_pred[:1].float(), gt.float())
            loss_dict['vico_loss'] = vico_loss
            parsed_loss, log_vars = self.parse_losses(loss_dict)
            optim_wrapper.update_params(parsed_loss)

        return log_vars

    @torch.no_grad()
    def infer(self,
              prompt: Union[str, List[str]],
              image_reference: Image.Image = None,
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
        text_embeddings = self._encode_prompt(prompt, device,
                                              num_images_per_prompt,
                                              do_classifier_free_guidance,
                                              negative_prompt)
        uncond_embeddings, text_embeddings = text_embeddings.chunk(2)
        uncond_embeddings = torch.cat([uncond_embeddings] * 2)
        text_embeddings = torch.cat([text_embeddings] * 2)
        ph_tokens = self.tokenizer(
            num_images_per_prompt * [self.placeholder],
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True)['input_ids']
        input_ids = self.tokenizer(
            num_images_per_prompt * prompt,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True)['input_ids']
        ph_tok = ph_tokens[0, 1]
        # TODO fix hard code
        clip_eot_token_id = 49407
        endoftext_idx = (torch.arange(input_ids.shape[0]),
                         torch.nonzero(input_ids == clip_eot_token_id)
                         [:batch_size, 1].repeat(num_images_per_prompt))
        placeholder_idx = torch.where(input_ids == ph_tok)
        if self.placeholder in prompt[0]:
            ph_pos = [placeholder_idx, endoftext_idx]
        else:
            ph_pos = [endoftext_idx, endoftext_idx]

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
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        image_reference = self.prepare_reference(
            image_reference,
            height,
            width,
        )
        image_reference = self.vae.encode(
            image_reference.to(dtype=img_dtype,
                               device=device)).latent_dist.sample()
        image_reference = image_reference.expand(
            batch_size * num_images_per_prompt, -1, -1, -1)
        image_reference = image_reference * self.vae.config.scaling_factor

        # 6. Prepare extra step kwargs.
        # TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        if show_progress:
            timesteps = tqdm(timesteps)
        for i, t in enumerate(timesteps):
            latents = torch.cat([latents, image_reference], dim=0)
            latent_model_input = self.test_scheduler.scale_model_input(
                latents, t)
            latent_model_input = latent_model_input.to(latent_dtype)
            text_embeddings = text_embeddings.to(latent_dtype)
            uncond_embeddings = uncond_embeddings.to(latent_dtype)
            # predict the noise residual

            noise_pred_uncond = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_embeddings,
                placeholder_position=ph_pos)['sample'][:batch_size *
                                                       num_images_per_prompt]
            noise_pred_text = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                placeholder_position=ph_pos)['sample'][:batch_size *
                                                       num_images_per_prompt]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_to_denoise = latents[:batch_size *
                                             num_images_per_prompt, ...]
                latents = self.test_scheduler.step(
                    noise_pred, t, latents_to_denoise,
                    **extra_step_kwargs)['prev_sample']

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

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """forward is not implemented now."""
        raise NotImplementedError(
            'Forward is not implemented now, please use infer.')
