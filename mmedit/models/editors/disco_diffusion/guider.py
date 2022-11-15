import torch.nn as nn 
import torchvision.transforms as T
import torch
import torch.nn.functional as F

from .prompt_utils import normalize
from mmedit.registry import MODULES
import pandas as pd
import numpy as np
from mmedit.models.losses import spherical_dist_loss, tv_loss, range_loss
from argparse import Namespace
import clip
import math
from resize_right import resize
import torchvision.transforms.functional as TF
import lpips
from .secondary_model import *

def sinc(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    """_summary_

    Args:
        x (_type_): _description_
        a (_type_): _description_

    Returns:
        _type_: _description_
    """
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
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
    """_summary_

    Args:
        input (_type_): _description_
        size (_type_): _description_
        align_corners (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
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
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
    """_summary_

    Args:
        cut_size (_type_): _description_
        cutn (_type_): _description_
        skip_augs (bool, optional): _description_. Defaults to False.
    """
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2]//4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn//4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts

cutout_debug = False

class MakeCutoutsDango(nn.Module):
    """_summary_

    Args:
        cut_size (_type_): _description_
        Overview (int, optional): _description_. Defaults to 4.
        InnerCrop (int, optional): _description_. Defaults to 0.
        IC_Size_Pow (float, optional): _description_. Defaults to 0.5.
        IC_Grey_P (float, optional): _description_. Defaults to 0.2.
    """
    def __init__(self, cut_size,
                 Overview=4, 
                 InnerCrop = 0, IC_Size_Pow=0.5, IC_Grey_P = 0.2
                 ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P

        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

          

    def forward(self, input, skip_augs = False):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1,3,self.cut_size,self.cut_size] 
        output_shape_2 = [1,3,self.cut_size+2,self.cut_size+2]
        pad_input = F.pad(input,((sideY-max_size)//2,(sideY-max_size)//2,(sideX-max_size)//2,(sideX-max_size)//2), **padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview>0:
            if self.Overview<=4:
                if self.Overview>=1:
                    cutouts.append(cutout)
                if self.Overview>=2:
                    cutouts.append(gray(cutout))
                if self.Overview>=3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview==4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

                              
        if self.InnerCrop >0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True: cutouts=self.augs(cutouts)
        return cutouts

def parse_prompt(prompt):
    """_summary_

    Args:
        prompt (_type_): _description_

    Returns:
        _type_: _description_
    """
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

def split_prompts(prompts, max_frames=1):
    """_summary_

    Args:
        prompts (_type_): _description_
        max_frames (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series


@MODULES.register_module()
class ImageTextGuider(nn.Module):
    """_summary_

    Args:
        clip_models (_type_): _description_
        cutter_cfg (_type_): _description_
        loss_cfg (_type_): _description_
    """
    def __init__(self, clip_models, cutter_cfg, loss_cfg):
        super().__init__()
        self.clip_models = clip_models
        self.cutter_cfg = Namespace(**cutter_cfg)
        # loss cfg
        self.loss_cfg = Namespace(**loss_cfg)
        self.lpips_model = lpips.LPIPS(net='vgg')

    
    def frame_prompt_from_text(self, text_prompts, frame_num=0):
        prompts_series = split_prompts(text_prompts) 
        if prompts_series is not None and frame_num >= len(prompts_series):
            frame_prompt = prompts_series[-1]
        elif prompts_series is not None:
            frame_prompt = prompts_series[frame_num]
        else:
            frame_prompt = []
        return frame_prompt

    def compute_prompt_stats(self, cutn=16, text_prompts=[], image_prompt=None, fuzzy_prompt=False, rand_mag=0.05):
        model_stats = []
        frame_prompt = self.frame_prompt_from_text(text_prompts)
        for clip_model in self.clip_models:
            model_stat = {"clip_model":None,"target_embeds":[],"make_cutouts":None,"weights":[]}
            model_stat["clip_model"] = clip_model
            
            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.model.encode_text(clip.tokenize(prompt).to(self.device)).float()
                
                if fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append((txt + torch.randn(txt.shape).cuda() * rand_mag).clamp(0,1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)
        
            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=self.device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)
        return model_stats

    def cond_fn(self, model, diffuser, x, t, beta_prod_t, model_stats, secondary_model=None, init_image=None, y=None, clamp_grad=True, clamp_max = 0.05, clip_guidance_scale=5000):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]
            if secondary_model is not None:
                alpha = torch.tensor(diffuser.alphas_cumprod[t]**0.5, dtype=torch.float32)
                sigma = torch.tensor((1 - diffuser.alphas_cumprod[t]) ** 0.5, dtype=torch.float32)
                cosine_t = alpha_sigma_to_t(alpha, sigma).to(x.device)
                model_output = secondary_model(x, cosine_t[None].repeat([x.shape[0]]))
                pred_original_sample = model_output['pred']
            else:
                model_output = model(x, t)['outputs']
                model_output, predicted_variance = torch.split(model_output, x.shape[1], dim=1)
                alpha_prod_t = 1 - beta_prod_t
                pred_original_sample = (x - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            # fac = diffuser_output['beta_prod_t']** (0.5)
            # x_in = diffuser_output['original_sample'] * fac + x * (1 - fac)
            fac = beta_prod_t** (0.5)
            x_in = pred_original_sample * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
            for model_stat in model_stats:
                for i in range(self.cutter_cfg.cutn_batches):
                    t_int = int(t.item())+1 #errors on last step without +1, need to find source
                    #when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                    try:
                        input_resolution=model_stat["clip_model"].model.visual.input_resolution
                    except:
                        input_resolution=224

                    cuts = MakeCutoutsDango(input_resolution,
                            Overview= self.cutter_cfg.cut_overview[1000-t_int], 
                            InnerCrop = self.cutter_cfg.cut_innercut[1000-t_int],
                            IC_Size_Pow= self.cutter_cfg.cut_ic_pow[1000-t_int],
                            IC_Grey_P = self.cutter_cfg.cut_icgray_p[1000-t_int]
                            )
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = model_stat["clip_model"].model.encode_image(clip_in).float()
                    dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                    dists = dists.view([self.cutter_cfg.cut_overview[1000-t_int]+self.cutter_cfg.cut_innercut[1000-t_int], n, -1])
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    # loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / self.cutter_cfg.cutn_batches
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(pred_original_sample)
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
            loss = tv_losses.sum() * self.loss_cfg.tv_scale + range_losses.sum() * self.loss_cfg.range_scale + sat_losses.sum() * self.loss_cfg.sat_scale
            if init_image is not None and self.loss_cfg.init_scale:
                init_losses = self.lpips_model(x_in, init_image)
                loss = loss + init_losses.sum() * self.loss_cfg.init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if torch.isnan(x_in_grad).any()==False:
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                # print("NaN'd")
                x_is_NaN = True
                grad = torch.zeros_like(x)
        if clamp_grad and x_is_NaN == False:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(max=clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
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