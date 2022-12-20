# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import numpy as np
import torch

from mmedit.models.utils.diffusion_utils import betas_for_alpha_bar
from mmedit.registry import DIFFUSION_SCHEDULERS


@DIFFUSION_SCHEDULERS.register_module()
class DDIMScheduler:
    """```DDIMScheduler``` support the diffusion and reverse process formulated
    in https://arxiv.org/abs/2010.02502.

    The code is heavily influenced by https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py. # noqa
    The difference is that we ensemble gradient-guided sampling in step function.

    Args:
        num_train_timesteps (int, optional): _description_. Defaults to 1000.
        beta_start (float, optional): _description_. Defaults to 0.0001.
        beta_end (float, optional): _description_. Defaults to 0.02.
        beta_schedule (str, optional): _description_. Defaults to "linear".
        variance_type (str, optional): _description_. Defaults to 'learned_range'.
        timestep_values (_type_, optional): _description_. Defaults to None.
        clip_sample (bool, optional): _description_. Defaults to True.
        set_alpha_to_one (bool, optional): _description_. Defaults to True.
    """

    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='linear',
        variance_type='learned_range',
        timestep_values=None,
        clip_sample=True,
        set_alpha_to_one=True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.timestep_values = timestep_values
        self.clip_sample = clip_sample
        self.set_alpha_to_one = set_alpha_to_one

        if beta_schedule == 'linear':
            self.betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=np.float32)
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

        # At every step in ddim, we are looking into the
        # previous alphas_cumprod. For the final step,
        # there is no previous alphas_cumprod because we are already
        # at 0 `set_alpha_to_one` decides whether we set this paratemer
        # simply to one or whether we use the final alpha of the
        # "non-previous" one.
        self.final_alpha_cumprod = np.array(
            1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

    def set_timesteps(self, num_inference_steps, offset=0):
        """set time steps."""

        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(
            0, self.num_train_timesteps,
            self.num_train_timesteps // self.num_inference_steps)[::-1].copy()
        self.timesteps += offset

    def _get_variance(self, timestep, prev_timestep):
        """get variance."""

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev /
                    beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        cond_fn=None,
        cond_kwargs={},
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
    ):
        """step forward."""

        output = {}
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is 'None', '\
                    'you need to run 'set_timesteps' '\
                        'after creating the scheduler")

        pred = None
        if isinstance(model_output, dict):
            pred = model_output['pred']
            model_output = model_output['eps']
        elif model_output.shape[1] == sample.shape[
                1] * 2 and self.variance_type in ['learned', 'learned_range']:
            model_output, _ = torch.split(model_output, sample.shape[1], dim=1)
        else:
            if not model_output.shape == sample.shape:
                raise TypeError

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf # noqa
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointingc to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep - self.num_train_timesteps // self.num_inference_steps)

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf # noqa
        pred_original_sample = (sample - (
            (beta_prod_t)**(0.5)) * model_output) / alpha_prod_t**(0.5)
        if pred is not None:
            pred_original_sample = pred

        gradient = 0.
        if cond_fn is not None:
            gradient = cond_fn(
                cond_kwargs.pop('unet'), self, sample, timestep, beta_prod_t,
                cond_kwargs.pop('model_stats'), **cond_kwargs)
            model_output = model_output - (beta_prod_t**0.5) * gradient
            pred_original_sample = (
                sample -
                (beta_prod_t**(0.5)) * model_output) / alpha_prod_t**(0.5)
        # 4. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance**(0.5)
        output.update(dict(sigma=std_dev_t))

        if use_clipped_model_output:
            # the model_output is always
            # re-derived from the clipped x_0 in Glide
            model_output = (sample - (alpha_prod_t**(0.5)) *
                            pred_original_sample) / beta_prod_t**(0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf # noqa
        pred_sample_direction = (1 - alpha_prod_t_prev -
                                 std_dev_t**2)**(0.5) * model_output

        # 7. compute x_t without "random noise" of
        # formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_mean = alpha_prod_t_prev**(
            0.5) * pred_original_sample + pred_sample_direction
        output.update(dict(mean=prev_mean, prev_sample=prev_mean))

        if eta > 0:
            device = model_output.device if torch.is_tensor(
                model_output) else 'cpu'
            noise = torch.randn(
                model_output.shape, generator=generator).to(device)
            variance = std_dev_t * noise

            if not torch.is_tensor(model_output):
                variance = variance.numpy()

            prev_sample = prev_mean + variance
            output.update({'prev_sample': prev_sample})

        # NOTE: this x0 is twice computed
        output.update({
            'original_sample': pred_original_sample,
            'beta_prod_t': beta_prod_t
        })
        return output

    def add_noise(self, original_samples, noise, timesteps):
        """add noise."""

        sqrt_alpha_prod = self.alphas_cumprod[timesteps]**0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps])**0.5
        noisy_samples = (
            sqrt_alpha_prod * original_samples +
            sqrt_one_minus_alpha_prod * noise)
        return noisy_samples

    def __len__(self):
        return self.num_train_timesteps
