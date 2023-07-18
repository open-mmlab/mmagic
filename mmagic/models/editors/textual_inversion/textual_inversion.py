# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MMLogger

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList
from ..stable_diffusion.stable_diffusion import StableDiffusion

logger = MMLogger.get_current_instance()

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class TextualInversion(StableDiffusion):
    """Implementation of `An Image is Worth One Word: Personalizing Text-to-
    Image Generation using Textual Inversion.

    <https://arxiv.org/abs/2208.01618>`_ (Textual Inversion).

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
        initialize_token (str, optional): The initialization token for textual
            embedding to train. Defaults to None.
        num_vefctor_per_token (int): The length of the learnable embedding.
            Defaults to 1.
        val_prompts (Union[str, List[str]], optional): The prompts for
            validation. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Defaults to
                dict(type='DataPreprocessor').
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`. Defaults to None/
    """

    def __init__(self,
                 placeholder_token: str,
                 vae: ModelType,
                 text_encoder: ModelType,
                 tokenizer: str,
                 unet: ModelType,
                 scheduler: ModelType,
                 test_scheduler: Optional[ModelType] = None,
                 dtype: Optional[str] = None,
                 enable_xformers: bool = True,
                 noise_offset_weight: float = 0,
                 tomesd_cfg: Optional[dict] = None,
                 initialize_token: Optional[str] = None,
                 num_vectors_per_token: int = 1,
                 val_prompts=None,
                 data_preprocessor: Optional[ModelType] = dict(
                     type='DataPreprocessor'),
                 init_cfg: Optional[dict] = None):

        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         test_scheduler, dtype, enable_xformers,
                         noise_offset_weight, tomesd_cfg, data_preprocessor,
                         init_cfg)

        self.val_prompts = val_prompts
        self.placeholder_token = placeholder_token
        self.add_tokens(placeholder_token, initialize_token,
                        num_vectors_per_token)
        self.prepare_models()

    def prepare_models(self):
        """Disable gradient for untrainable modules to save memory."""
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.set_only_embedding_trainable()

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

        unet_dtype = next(self.unet.parameters()).dtype
        self.unet.to(self.dtype)

        output = self.infer(prompt, return_type='tensor')
        samples = output['samples']

        self.unet.to(unet_dtype)

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

        unet_dtype = next(self.unet.parameters()).dtype
        self.unet.to(self.dtype)

        output = self.infer(prompt, return_type='tensor')
        samples = output['samples']

        self.unet.to(unet_dtype)

        samples = self.data_preprocessor.destruct(samples, data_samples)

        out_data_sample = DataSample(fake_img=samples, prompt=prompt)
        data_sample_list = out_data_sample.split()
        return data_sample_list

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

    def train_step(self, data, optim_wrapper):
        """Training step."""
        data = self.data_preprocessor(data)
        inputs, data_samples = data['inputs'], data['data_samples']

        vae = self.vae.module if hasattr(self.vae, 'module') else self.vae
        vae_dtype = next(vae.parameters()).dtype
        unet_dtype = next(self.unet.parameters()).dtype

        with optim_wrapper.optim_context(self.unet):
            image = inputs  # image for new concept
            prompt = data_samples.prompt
            num_batches = image.shape[0]

            image = image.to(vae_dtype)
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

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

            model_output = self.unet(
                noisy_latents.to(unet_dtype),
                timesteps,
                encoder_hidden_states=encoder_hidden_states.to(unet_dtype))
            model_pred = model_output['sample']

            loss_dict = dict()

            # calculate loss in FP32
            loss = F.mse_loss(model_pred.float(), gt.float())
            loss_dict['loss'] = loss

            parsed_loss, log_vars = self.parse_losses(loss_dict)
            optim_wrapper.update_params(parsed_loss)

        return log_vars
