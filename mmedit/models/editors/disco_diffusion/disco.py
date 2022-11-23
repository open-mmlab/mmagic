# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union

import mmcv
import mmengine
import torch
import torch.nn as nn
from mmengine.runner import set_random_seed
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_with_prefix)
from tqdm import tqdm

from mmedit.registry import DIFFUSION_SCHEDULERS, MODELS, MODULES
from .guider import ImageTextGuider

ModelType = Union[Dict, nn.Module]


@MODELS.register_module('disco')
@MODELS.register_module('dd')
@MODELS.register_module()
class DiscoDiffusion(nn.Module):
    """Disco Diffusion (DD) is a Google Colab Notebook which leverages an AI
    Image generating technique called CLIP-Guided Diffusion to allow you to
    create compelling and beautiful images from just text inputs. Created by
    Somnai, augmented by Gandamu, and building on the work of RiversHaveWings,
    nshepperd, and many others.
    
    Ref:
        Github Repo: https://github.com/alembics/disco-diffusion
        Colab: https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb # noqa
        
    Args:
        unet (ModelType): Config of denoising Unet.
        diffuser (ModelType): Config of diffuser scheduler.
        secondary_model (ModelType): There are two diff. Defaults to None.
        cutter_cfg (dict): The config of image cutter. Defaults to dict().
        loss_cfg (dict): Loss config for computing gradient. Defaults to
            dict().
        clip_models_cfg (list): Config of clip models. Defaults to [].
        use_fp16 (bool): Whether to use fp16 for unet model. Defaults to False.
        pretrained_cfgs (dict): Path Config for pretrained weights. Usually
            this is a dict contains module name and the corresponding ckpt
            path.Defaults to None.
    """

    def __init__(self,
                 unet,
                 diffuser,
                 secondary_model=None,
                 cutter_cfg=dict(),
                 loss_cfg=dict(),
                 clip_models_cfg=[],
                 use_fp16=False,
                 pretrained_cfgs=None):
        super().__init__()
        self.unet = MODULES.build(unet)
        self.diffuser = DIFFUSION_SCHEDULERS.build(diffuser)
        clip_models = []
        for clip_cfg in clip_models_cfg:
            clip_models.append(MODULES.build(clip_cfg))
        self.guider = ImageTextGuider(clip_models, cutter_cfg, loss_cfg)

        if secondary_model is not None:
            self.secondary_model = MODULES.build(secondary_model)
            self.with_secondary_model = True
        else:
            self.with_secondary_model = False

        if pretrained_cfgs:
            self.load_pretrained_models(pretrained_cfgs)
        if use_fp16:
            mmengine.print_log('Convert unet modules to floatpoint16')
            self.unet.convert_to_fp16()

    def load_pretrained_models(self, pretrained_cfgs):
        """Loading pretrained weights to model. ``pretrained_cfgs`` is a dict
        consist of module name as key and checkpoint path as value.

        Args:
            pretrained_cfgs (dict): Path Config for pretrained weights.
            Usually this is a dict contains module name and the
            corresponding ckpt path. Defaults to None.
        """
        for key, ckpt_cfg in pretrained_cfgs.items():
            prefix = ckpt_cfg.get('prefix', '')
            map_location = ckpt_cfg.get('map_location', 'cpu')
            strict = ckpt_cfg.get('strict', True)
            ckpt_path = ckpt_cfg.get('ckpt_path')
            if prefix:
                state_dict = _load_checkpoint_with_prefix(
                    prefix, ckpt_path, map_location)
            else:
                state_dict = _load_checkpoint(ckpt_path, map_location)
            getattr(self, key).load_state_dict(state_dict, strict=strict)
            mmengine.print_log(f'Load pretrained {key} from {ckpt_path}')

    @property
    def device(self):
        """Get current device of the model.

        Returns:
            torch.device: The current device of the model.
        """
        return next(self.parameters()).device

    @torch.no_grad()
    def infer(self,
              scheduler_kwargs=None,
              height=None,
              width=None,
              init_image=None,
              batch_size=1,
              num_inference_steps=1000,
              skip_steps=0,
              show_progress=False,
              text_prompts=[],
              image_prompts=[],
              eta=0.8,
              clip_guidance_scale=5000,
              seed=None):
        """Inference API for disco diffusion.

        Args:
            scheduler_kwargs (dict): Args for infer time diffusion
                scheduler. Defaults to None.
            height (int): Height of output image. Defaults to None.
            width (int): Width of output image. Defaults to None.
            init_image (str): Initial image at the start point
                of denoising. Defaults to None.
            batch_size (int): Batch size. Defaults to 1.
            num_inference_steps (int): Number of inference steps.
                Defaults to 1000.
            skip_steps (int): Denoising steps to skip, usually set
                with ``init_image``. Defaults to 0.
            show_progress (bool): Whether to show progress.
                Defaults to False.
            text_prompts (list): Text prompts. Defaults to [].
            image_prompts (list): Image prompts, this is not the same as
                ``init_image``, they works the same way with
                ``text_prompts``. Defaults to [].
            eta (float): Eta for ddim sampling. Defaults to 0.8.
            clip_guidance_scale (int): The Scale of influence of prompts
                on output image. Defaults to 1000.
            seed (int): Sampling seed. Defaults to None.
        """
        # set diffuser
        if scheduler_kwargs is not None:
            mmengine.print_log('Switch to infer diffusion scheduler!',
                               'current')
            infer_scheduler = DIFFUSION_SCHEDULERS.build(scheduler_kwargs)
        else:
            infer_scheduler = self.diffuser
        # set random seed
        if isinstance(seed, int):
            set_random_seed(seed=seed)

        # set step values
        if num_inference_steps > 0:
            infer_scheduler.set_timesteps(num_inference_steps)

        _ = image_prompts

        height = (height // 64) * 64 if height else self.unet.image_size
        width = (width // 64) * 64 if width else self.unet.image_size
        if init_image is None:
            image = torch.randn(
                (batch_size, self.unet.in_channels, height, width))
            image = image.to(self.device)
        else:
            init = mmcv.imread(init_image, channel_order='rgb')
            init = mmcv.imresize(
                init, (width, height), interpolation='lanczos') / 255.
            init_image = torch.as_tensor(
                init,
                dtype=torch.float32).to(self.device).unsqueeze(0).permute(
                    0, 3, 1, 2).mul(2).sub(1)
            image = init_image.clone()
            image = infer_scheduler.add_noise(
                image, torch.randn_like(image),
                infer_scheduler.timesteps[skip_steps])
        # get stats from text prompts and image prompts
        model_stats = self.guider.compute_prompt_stats(
            text_prompts=text_prompts)
        timesteps = infer_scheduler.timesteps[skip_steps:-1]
        if show_progress:
            timesteps = tqdm(timesteps)
        for t in timesteps:
            # 1. predicted model_output
            model_output = self.unet(image, t)['outputs']

            # 2. compute previous image: x_t -> x_t-1
            cond_kwargs = {
                'model_stats': model_stats,
                'init_image': init_image,
                'unet': self.unet,
                'clip_guidance_scale':clip_guidance_scale
            }
            if self.with_secondary_model:
                cond_kwargs.update(secondary_model=self.secondary_model)
            diffuser_output = infer_scheduler.step(
                model_output,
                t,
                image,
                cond_fn=self.guider.cond_fn,
                cond_kwargs=cond_kwargs,
                eta=eta)

            image = diffuser_output['prev_sample']
        return {'samples': image}