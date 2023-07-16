"""This code was originally taken from https://github.com/google/prompt-to-
prompt and modified by Ferry."""

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import numpy as np
import torch
from PIL import Image

from mmagic.models.diffusion_schedulers import EditDDIMScheduler


class DDIMInversion:

    def __init__(self, model, **kwargs):
        scheduler = EditDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False)
        self.model = model
        self.num_ddim_steps = kwargs.pop('num_ddim_steps', 50)
        self.guidance_scale = kwargs.pop('guidance_scale', 7.5)
        self.tokenizer = self.model.tokenizer
        self.model.scheduler = scheduler
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - (
            self.scheduler.num_train_timesteps //
            self.scheduler.num_inference_steps)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev)**0.5 * model_output
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample \
            + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.num_train_timesteps //
            self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (
            sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next)**0.5 * model_output
        next_sample = alpha_prod_t_next**0.5 * next_original_sample \
            + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(
            latents, t, encoder_hidden_states=context)['sample']
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(
            latents_input, t, encoder_hidden_states=context)['sample']
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents, return_dict=False)[0]
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0,
                                      1).unsqueeze(0).to(self.model.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [''],
            padding='max_length',
            max_length=self.model.tokenizer.model_max_length,
            return_tensors='pt')
        uncond_embeddings = self.model.text_encoder(
            uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding='max_length',
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_embeddings = self.model.text_encoder(
            text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.model.vae.encode(image)['latent_dist'].mean
        latent = latent * 0.18215
        # image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return ddim_latents
