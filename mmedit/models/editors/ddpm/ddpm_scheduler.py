# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import numpy as np
import torch

from mmedit.registry import DIFFUSION_SCHEDULERS
from ...utils.diffusion_utils import betas_for_alpha_bar


@DIFFUSION_SCHEDULERS.register_module()
class DDPMScheduler:

    def __init__(self,
                 num_train_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule='linear',
                 trained_betas=None,
                 variance_type='fixed_small',
                 clip_sample=True):
        """```DDPMScheduler``` support the diffusion and reverse process
        formulated in https://arxiv.org/abs/2006.11239.

        The code is heavily influenced by https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py. # noqa

        Args:
            num_train_timesteps (int, optional): _description_. Defaults to 1000.
            beta_start (float, optional): _description_. Defaults to 0.0001.
            beta_end (float, optional): _description_. Defaults to 0.02.
            beta_schedule (str, optional): _description_. Defaults to 'linear'.
            trained_betas (_type_, optional): _description_. Defaults to None.
            variance_type (str, optional): _description_. Defaults to 'fixed_small'.
            clip_sample (bool, optional): _description_. Defaults to True.
        """
        self.num_train_timesteps = num_train_timesteps
        if trained_betas is not None:
            self.betas = np.asarray(trained_betas)
        elif beta_schedule == 'linear':
            self.betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        elif beta_schedule == 'scaled_linear':
            # this schedule is very specific to the latent diffusion model.
            self.betas = np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_train_timesteps,
                dtype=np.float32)**2
        elif beta_schedule == 'squaredcos_cap_v2':
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(
                f'{beta_schedule} does is not implemented for {self.__class__}'
            )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.one = np.array(1.0)

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

        self.variance_type = variance_type
        self.clip_sample = clip_sample

    def set_timesteps(self, num_inference_steps):
        num_inference_steps = min(self.num_train_timesteps,
                                  num_inference_steps)
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(
            0, self.num_train_timesteps,
            self.num_train_timesteps // self.num_inference_steps)[::-1].copy()

    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf) # noqa
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample # noqa
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]

        if t == 0:
            log_variance = (1 - alpha_prod_t_prev) / (
                1 - alpha_prod_t) * self.betas[1]
        else:
            log_variance = np.log(variance)

        if variance_type is None:
            variance_type = self.variance_type

        # hacks - were probs added for training stability
        if variance_type == 'fixed_small':
            variance = np.clip(variance, min_value=1e-20)
        # for rl-diffusion_scheduler https://arxiv.org/abs/2205.09991
        elif variance_type == 'fixed_small_log':
            variance = np.log(np.clip(variance, min_value=1e-20))
        elif variance_type == 'fixed_large':
            variance = self.betas[t]
        elif variance_type == 'fixed_large_log':
            # Glide max_log
            variance = np.log(self.betas[t])
        elif variance_type == 'learned':
            return predicted_variance
        elif variance_type == 'learned_range':
            min_log = log_variance
            max_log = np.log(self.betas[t])
            frac = (predicted_variance + 1) / 2
            log_variance = frac * max_log + (1 - frac) * min_log
            variance = torch.exp(log_variance)

        return variance

    def step(self,
             model_output: Union[torch.FloatTensor],
             timestep: int,
             sample: Union[torch.FloatTensor],
             predict_epsilon=True,
             generator=None):
        t = timestep

        if model_output.shape[1] == sample.shape[
                1] * 2 and self.variance_type in ['learned', 'learned_range']:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf # noqa
        if predict_epsilon:
            pred_original_sample = (
                (sample - beta_prod_t**(0.5) * model_output) /
                alpha_prod_t**(0.5))
        else:
            pred_original_sample = model_output

        # 3. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t # noqa
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev**(0.5) *
                                      self.betas[t]) / beta_prod_t
        current_sample_coeff = self.alphas[t]**(
            0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_mean = (
            pred_original_sample_coeff * pred_original_sample +
            current_sample_coeff * sample)

        # 6. Add noise
        noise = torch.randn_like(model_output)
        sigma = 0
        if t > 0:
            sigma = self._get_variance(
                t, predicted_variance=predicted_variance)**0.5

        pred_prev_sample = pred_prev_mean + sigma * noise

        return {
            'prev_sample': pred_prev_sample,
            'mean': pred_prev_mean,
            'sigma': sigma,
            'noise': noise
        }

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.alphas_cumprod[timesteps]**0.5
        sqrt_alpha_prod = self.match_shape(sqrt_alpha_prod, original_samples)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps])**0.5
        sqrt_one_minus_alpha_prod = self.match_shape(sqrt_one_minus_alpha_prod,
                                                     original_samples)

        noisy_samples = (
            sqrt_alpha_prod * original_samples +
            sqrt_one_minus_alpha_prod * noise)
        return noisy_samples

    def training_loss(self, model, x_0, t):
        raise NotImplementedError(
            'This function is supposed to return '
            'a dict containing loss items giving sampled x0 and timestep.')

    def sample_timestep(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_train_timesteps
