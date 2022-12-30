# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmedit.models.editors.stable_diffusion import StableDiffusion
from mmedit.registry import MODULES


@MODULES.register_module()
class StableDiffusionWrapper(StableDiffusion):
    """Stable diffusion wrapper for dreamfusion."""

    def __init__(self,
                 diffusion_scheduler,
                 unet_cfg,
                 vae_cfg,
                 pretrained_ckpt_path,
                 requires_safety_checker=True,
                 unet_sample_size=64):
        super().__init__(diffusion_scheduler, unet_cfg, vae_cfg,
                         pretrained_ckpt_path, requires_safety_checker,
                         unet_sample_size)
        self.min_step = int(0.02 * self.scheduler.num_train_timesteps)
        self.max_step = int(0.98 * self.scheduler.num_train_timesteps)

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.execution_device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.execution_device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    # @torch.no_grad()
    def decode_latents(self, latents):
        # TODO: can we do this by super().decode(latents) ?
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def train_step_(self, text_embeddings, pred_rgb, guidance_scale=100):

        text_embeddings = text_embeddings.to(self.execution_device)
        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(
            pred_rgb, (512, 512), mode='bilinear', align_corners=False)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1, [1],
            dtype=torch.long,
            device=self.execution_device)

        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=text_embeddings)['outputs']

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        # w = (1 - self.alphas[t])
        w = (1 - self.scheduler.alphas_cumprod[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot
        # simply autodiff.
        latents.backward(gradient=grad, retain_graph=True)

        # TODO: return a loss term without grad
        return 0  # dummy loss value
