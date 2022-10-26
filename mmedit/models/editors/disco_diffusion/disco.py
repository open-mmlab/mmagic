# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import List, Optional

import mmengine
import torch
from mmengine import MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapperDict
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix, _load_checkpoint
from tqdm import tqdm

from mmedit.registry import DIFFUSERS, MODELS, MODULES
from mmedit.structures import EditDataSample, PixelData
from mmedit.utils.typing import ForwardInputs, SampleList
from mmengine.runner import set_random_seed
from .guider import ImageTextGuider

from torchvision.utils import save_image
from .secondary_model import SecondaryDiffusionImageNet2, alpha_sigma_to_t

@MODELS.register_module('disco')
@MODELS.register_module('dd')
@MODELS.register_module()
class DiscoDiffusion(BaseModel):
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
        self.diffuser = DIFFUSERS.build(diffuser)
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
                state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                      map_location)
            else:
                state_dict = _load_checkpoint(ckpt_path,
                                                      map_location)
            getattr(self, key).load_state_dict(state_dict, strict=strict)
            mmengine.print_log(f'Load pretrained {key} from {ckpt_path}')

    @property
    def device(self):
        """Get current device of the model.

        Returns:
            torch.device: The current device of the model.
        """
        return next(self.parameters()).device



    def infer(self,
              height=None,
              width=None,
              init_image=None,
              batch_size=1,
              num_inference_steps=1000,
              show_progress=False,
              text_prompts=[],
              image_prompts=[],
              eta = 0.8,
              seed=None):
        """_summary_

        Args:
            init_image (_type_, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 1.
            num_inference_steps (int, optional): _description_.
                Defaults to 1000.
            labels (_type_, optional): _description_. Defaults to None.
            show_progress (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        height = (height//64)*64 if height else self.unet.image_size
        width = (width//64)*64 if width else self.unet.image_size
        # TODO: print modified height and width

        if init_image is None:
            image = torch.randn((batch_size, self.unet.in_channels,
                                 height, width))
            image = image.to(self.device)
        else:
            # TODO: resize init_image
            image = init_image

        loss_values = []
        # set random seed
        if isinstance(seed, int):
            set_random_seed(seed=seed)

        # get stats from text prompts and image prompts
        model_stats = self.guider.compute_prompt_stats(text_prompts=text_prompts)
        # set step values
        if num_inference_steps > 0:
            self.diffuser.set_timesteps(num_inference_steps)

        timesteps = self.diffuser.timesteps
        if show_progress:
            timesteps = tqdm(timesteps)
        for t in timesteps:
            # 1. predicted model_output
            model_output = self.unet(image, t)['outputs']

            # 2. compute previous image: x_t -> x_t-1
            cond_kwargs = {'model_stats': model_stats, 'init_image':init_image, 'unet': self.unet}
            if self.with_secondary_model:
                cond_kwargs.update(secondary_model=self.secondary_model)
            diffuser_output = self.diffuser.step(model_output, t, image, cond_fn=self.guider.cond_fn, cond_kwargs=cond_kwargs, eta = eta)

            image = diffuser_output['prev_sample']
            save_image(image, f"work_dirs/disco/{t}.png", normalize=True)

        return {'samples': image}

    def forward(self, x):
        return x

