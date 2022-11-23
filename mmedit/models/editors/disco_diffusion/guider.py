# Copyright (c) OpenMMLab. All rights reserved.
import math
from argparse import Namespace

import clip
import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from resize_right import resize

from mmedit.models.losses import range_loss, spherical_dist_loss, tv_loss
from .prompt_utils import normalize
from .secondary_model import alpha_sigma_to_t


def sinc(x):
    """
    Sinc function.
    If x equal to 0,
        sinc(x) = 1
    else:
        sinc(x) = sin(x)/ x
    Args:
        x (torch.Tensor): Input Tensor

    Returns:
        torch.Tensor: Function output.
    """
    return torch.where(x != 0,
                       torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    """Lanczos filter's reconstruction kernel L(x)."""
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    """_summary_

    Args:
        ratio (_type_): _description_
        width (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    """Lanczos resampling image.

    Args:
        input (torch.Tensor): Input image tensor.
        size (Tuple[int, int]): Output image size.
        align_corners (bool): align_corners argument of F.interpolate.
            Defaults to True.

    Returns:
        torch.Tensor: Resampling results.
    """
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(
        input, size, mode='bicubic', align_corners=align_corners)


class MakeCutouts(nn.Module):
    """Each iteration, the AI cuts the image into smaller pieces known as cuts
    , and compares each cut to the prompt to decide how to guide the next 
    diffusion step. 
    This classes will randomly cut patches and perfom image augmentation to 
    these patches.

    Args:
        cut_size (int): Size of the patches.
        cutn (int): Number of patches to cut.
    """

    def __init__(self, cut_size, cutn):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        ])

    def forward(self, input, skip_augs=False):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1, ).normal_(
                    mean=.8, std=.3).clip(float(self.cut_size / max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size,
                               offsetx:offsetx + size]

            if not skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts

class MakeCutoutsDango(nn.Module):
    """Dango233(https://github.com/Dango233)'s version of MakeCutouts.
    
    The improvement compared to ``MakeCutouts`` is that it use partial
    greyscale augmentation to capture structure, and partial rotation 
    augmentation to capture whole frames.
    
    Args:
        cut_size (int): Size of the patches.
        Overview (int): The total number of overview cuts.
        In details,
            Overview=1, Add whole frame;
            Overview=2, Add grayscaled frame;
            Overview=3, Add horizontal flip frame;
            Overview=4, Add grayscaled horizontal flip frame;
            Overview>4, Repeat add frame Overview times. 
            Defaults to 4.
        InnerCrop (int): The total number of inner cuts.
            Defaults to 0.
        IC_Size_Pow (float): This sets the size of the border
            used for inner cuts.  High values have larger borders,
            and therefore the cuts themselves will be smaller and 
            provide finer details. Defaults to 0.5.
        IC_Grey_P (float): The portion of the inner cuts can be set to be
            grayscale instead of color. This may help with improved 
            definition of shapes and edges, especially in the early
            diffusion steps where the image structure is being defined.
            Defaults to 0.2.
    """

    def __init__(self,
                 cut_size,
                 Overview=4,
                 InnerCrop=0,
                 IC_Size_Pow=0.5,
                 IC_Grey_P=0.2):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P

        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input, skip_augs=False):
        '''Forward function'''
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(input,
                          ((sideY - max_size) // 2, (sideY - max_size) // 2,
                           (sideX - max_size) // 2, (sideX - max_size) // 2))
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(
                    torch.rand([])**self.IC_Size_Pow * (max_size - min_size) +
                    min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size,
                               offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
        cutouts = torch.cat(cutouts)
        if not skip_augs:
            cutouts = self.augs(cutouts)
        return cutouts


def parse_prompt(prompt):
    """Parse prompt, return text and text weight"""
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


def split_prompts(prompts, max_frames=1):
    """Split prompts to a list of prompts."""
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series


class ImageTextGuider(nn.Module):
    """Disco-Diffusion uses text and images to guide image generation.
    We will use the clip models to extract text and image features as prompts,
    and then during the iteration, the features of the image patches are
    computed, and the similarity loss between the prompts features and the 
    generated features is computed. Other losses also include RGB Range loss,
    total variation loss. Using these losses we can guide the image generation 
    towards the desired target.

    Args:
        clip_models (List[Dict]): List of clip model settings.
        cutter_cfg (Dict): Setting of cutters during iteration.
        loss_cfg (Dict): Setting of losses.
    """

    def __init__(self, clip_models, cutter_cfg, loss_cfg):
        super().__init__()
        self.clip_models = clip_models
        self.cutter_cfg = Namespace(**cutter_cfg)
        # loss cfg
        self.loss_cfg = Namespace(**loss_cfg)
        self.lpips_model = lpips.LPIPS(net='vgg')

    def frame_prompt_from_text(self, text_prompts, frame_num=0):
        '''Get current frame prompt.'''
        prompts_series = split_prompts(text_prompts)
        if prompts_series is not None and frame_num >= len(prompts_series):
            frame_prompt = prompts_series[-1]
        elif prompts_series is not None:
            frame_prompt = prompts_series[frame_num]
        else:
            frame_prompt = []
        return frame_prompt

    def compute_prompt_stats(self,
                             cutn=16,
                             text_prompts=[],
                             image_prompt=None,
                             fuzzy_prompt=False,
                             rand_mag=0.05):
        """Compute prompts statistics. 

        Args:
            cutn (int, optional): _description_. Defaults to 16.
            text_prompts (list, optional): _description_. Defaults to [].
            image_prompt (_type_, optional): _description_. Defaults to None.
            fuzzy_prompt (bool, optional): _description_. Defaults to False.
            rand_mag (float, optional): _description_. Defaults to 0.05.
        """
        model_stats = []
        frame_prompt = self.frame_prompt_from_text(text_prompts)
        for clip_model in self.clip_models:
            model_stat = {
                'clip_model': None,
                'target_embeds': [],
                'make_cutouts': None,
                'weights': []
            }
            model_stat['clip_model'] = clip_model

            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.model.encode_text(
                    clip.tokenize(prompt).to(self.device)).float()

                if fuzzy_prompt:
                    for i in range(25):
                        model_stat['target_embeds'].append(
                            (txt +
                             torch.randn(txt.shape).cuda() * rand_mag).clamp(
                                 0, 1))
                        model_stat['weights'].append(weight)
                else:
                    model_stat['target_embeds'].append(txt)
                    model_stat['weights'].append(weight)

            model_stat['target_embeds'] = torch.cat(
                model_stat['target_embeds'])
            model_stat['weights'] = torch.tensor(
                model_stat['weights'], device=self.device)
            if model_stat['weights'].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat['weights'] /= model_stat['weights'].sum().abs()
            model_stats.append(model_stat)
        return model_stats

    def cond_fn(self,
                model,
                diffuser,
                x,
                t,
                beta_prod_t,
                model_stats,
                secondary_model=None,
                init_image=None,
                y=None,
                clamp_grad=True,
                clamp_max=0.05,
                clip_guidance_scale=5000):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]
            if secondary_model is not None:
                alpha = torch.tensor(
                    diffuser.alphas_cumprod[t]**0.5, dtype=torch.float32)
                sigma = torch.tensor(
                    (1 - diffuser.alphas_cumprod[t])**0.5, dtype=torch.float32)
                cosine_t = alpha_sigma_to_t(alpha, sigma).to(x.device)
                model_output = secondary_model(
                    x, cosine_t[None].repeat([x.shape[0]]))
                pred_original_sample = model_output['pred']
            else:
                model_output = model(x, t)['outputs']
                model_output, predicted_variance = torch.split(
                    model_output, x.shape[1], dim=1)
                alpha_prod_t = 1 - beta_prod_t
                pred_original_sample = (x - beta_prod_t**(0.5) *
                                        model_output) / alpha_prod_t**(0.5)
            # fac = diffuser_output['beta_prod_t']** (0.5)
            # x_in = diffuser_output['original_sample'] * fac + x * (1 - fac)
            fac = beta_prod_t**(0.5)
            x_in = pred_original_sample * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
            for model_stat in model_stats:
                for i in range(self.cutter_cfg.cutn_batches):
                    t_int = int(t.item()) + 1
                    try:
                        input_resolution = model_stat[
                            'clip_model'].model.visual.input_resolution
                    except AttributeError:
                        input_resolution = 224

                    cuts = MakeCutoutsDango(
                        input_resolution,
                        Overview=self.cutter_cfg.cut_overview[1000 - t_int],
                        InnerCrop=self.cutter_cfg.cut_innercut[1000 - t_int],
                        IC_Size_Pow=self.cutter_cfg.cut_ic_pow[1000 - t_int],
                        IC_Grey_P=self.cutter_cfg.cut_icgray_p[1000 - t_int])
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = model_stat['clip_model'].model.encode_image(
                        clip_in).float()
                    dists = spherical_dist_loss(
                        image_embeds.unsqueeze(1),
                        model_stat['target_embeds'].unsqueeze(0))
                    dists = dists.view([
                        self.cutter_cfg.cut_overview[1000 - t_int] +
                        self.cutter_cfg.cut_innercut[1000 - t_int], n, -1
                    ])
                    losses = dists.mul(model_stat['weights']).sum(2).mean(0)
                    x_in_grad += torch.autograd.grad(
                        losses.sum() * clip_guidance_scale,
                        x_in)[0] / self.cutter_cfg.cutn_batches
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(pred_original_sample)
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = tv_losses.sum() * self.loss_cfg.tv_scale + range_losses.sum(
            ) * self.loss_cfg.range_scale + sat_losses.sum(
            ) * self.loss_cfg.sat_scale
            if init_image is not None and self.loss_cfg.init_scale:
                init_losses = self.lpips_model(x_in, init_image)
                loss = loss + init_losses.sum() * self.loss_cfg.init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if not torch.isnan(x_in_grad).any():
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                x_is_NaN = True
                grad = torch.zeros_like(x)
        if clamp_grad and not x_is_NaN:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(max=clamp_max) / magnitude
        return grad

    @property
    def device(self):
        """Get current device of the model.

        Returns:
            torch.device: The current device of the model.
        """
        return next(self.parameters()).device

    def forward(self, x):
        return x

  