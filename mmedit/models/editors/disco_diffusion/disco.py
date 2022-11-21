# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import List, Optional

import mmcv
import mmengine
import torch
import torch.nn.functional as F
from mmengine import MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapperDict
from mmengine.runner import set_random_seed
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_with_prefix)
from torchvision.utils import save_image
from tqdm import tqdm

from mmedit.registry import DIFFUSION_SCHEDULERS, MODELS, MODULES
from mmedit.structures import EditDataSample, PixelData
from mmedit.utils.typing import ForwardInputs, SampleList
from .guider import ImageTextGuider
from .secondary_model import SecondaryDiffusionImageNet2, alpha_sigma_to_t


@MODELS.register_module('disco')
@MODELS.register_module('dd')
@MODELS.register_module()
class DiscoDiffusion(BaseModel):
    """Disco Diffusion (DD) is a Google Colab Notebook which leverages an AI
    Image generating technique called CLIP-Guided Diffusion to allow you to
    create compelling and beautiful images from just text inputs. Created by
    Somnai, augmented by Gandamu, and building on the work of RiversHaveWings,
    nshepperd, and many others.

    Args:
        data_preprocessor (_type_): _description_
        unet (_type_): _description_
        diffuser (_type_): _description_
        secondary_model (_type_, optional): _description_. Defaults to None.
        cutter_cfg (_type_, optional): _description_. Defaults to dict().
        loss_cfg (_type_, optional): _description_. Defaults to dict().
        clip_models_cfg (list, optional): _description_. Defaults to [].
        use_fp16 (bool, optional): _description_. Defaults to False.
        pretrained_cfgs (_type_, optional): _description_. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor,
                 unet,
                 diffuser,
                 secondary_model=None,
                 cutter_cfg=dict(),
                 loss_cfg=dict(),
                 clip_models_cfg=[],
                 use_fp16=False,
                 pretrained_cfgs=None):
        super().__init__(data_preprocessor=data_preprocessor)
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
        """_summary_

        Args:
            pretrained_cfgs (_type_): _description_
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
              clip_grad_scale=1000,
              seed=None):
        """_summary_

        Args:
            scheduler_kwargs (dict, optional):
            init_image (_type_, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 1.
            num_inference_steps (int, optional): _description_.
                Defaults to 1000.
            labels (_type_, optional): _description_. Defaults to None.
            show_progress (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
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
                'unet': self.unet
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

    def forward(self, inputs, data_samples, mode):
        raise NotImplementedError(
            "Disco Diffusion doesn't have forward function")
