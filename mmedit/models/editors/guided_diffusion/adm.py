# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import List, Optional

import mmengine
import torch
from mmengine.model import BaseModel
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from tqdm import tqdm

from mmedit.registry import DIFFUSERS, MODELS, MODULES
from mmedit.structures import EditDataSample, PixelData
from mmedit.utils.typing import ForwardInputs, SampleList


@MODELS.register_module('ADM')
@MODELS.register_module('GuidedDiffusion')
class AblatedDiffusionModel(BaseModel):

    def __init__(self,
                 data_preprocessor,
                 unet,
                 diffuser,
                 use_fp16=False,
                 classifier=None,
                 pretrained_cfgs=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.unet = MODULES.build(unet)
        self.diffuser = DIFFUSERS.build(diffuser)
        if classifier:
            self.classifier = MODULES.build(unet)
        if pretrained_cfgs:
            self.load_pretrained_models(pretrained_cfgs)
        if use_fp16:
            mmengine.print_log('Convert unet modules to floatpoint16')
            self.unet.convert_to_fp16()

    def load_pretrained_models(self, pretrained_cfgs):
        for key, ckpt_cfg in pretrained_cfgs.items():
            prefix = ckpt_cfg.get('prefix', '')
            map_location = ckpt_cfg.get('map_location', 'cpu')
            strict = ckpt_cfg.get('strict', True)
            ckpt_path = ckpt_cfg.get('ckpt_path')
            state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
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
              init_image=None,
              batch_size=1,
              num_inference_steps=1000,
              labels=None,
              show_progress=False):
        # Sample gaussian noise to begin loop
        if init_image is None:
            image = torch.randn((batch_size, self.unet.in_channels,
                                 self.unet.image_size, self.unet.image_size))
            image = image.to(self.device)
        else:
            image = init_image

        if isinstance(labels, int):
            labels = torch.tensor(labels).repeat(batch_size, 1)
        elif labels is None:
            labels = torch.randint(
                low=0,
                high=self.unet.num_classes,
                size=(batch_size, ),
                device=self.device)

        # set step values
        if num_inference_steps > 0:
            self.diffuser.set_timesteps(num_inference_steps)

        timesteps = self.diffuser.timesteps
        if show_progress:
            timesteps = tqdm(timesteps)
        for t in timesteps:
            # 1. predict noise model_output
            model_output = self.unet(image, t, label=labels)['outputs']

            # 2. compute previous image: x_t -> t_t-1
            image = self.diffuser.step(model_output, t, image)['prev_sample']

        return {'samples': image}

    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> List[EditDataSample]:
        init_image = inputs.get('init_image', None)
        batch_size = inputs.get('batch_size', 1)
        labels = data_samples.get('labels', None)
        sample_kwargs = inputs.get('sample_kwargs', dict())

        num_inference_steps = sample_kwargs.get(
            'num_inference_steps', self.diffuser.num_train_timesteps)
        show_progress = sample_kwargs.get('show_progress', False)

        outputs = self.infer(
            init_image=init_image,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            show_progress=show_progress)

        batch_sample_list = []
        for idx in range(batch_size):
            gen_sample = EditDataSample()
            if data_samples:
                gen_sample.update(data_samples[idx])
            if isinstance(outputs, dict):
                gen_sample.ema = EditDataSample(
                    fake_img=PixelData(data=outputs['ema'][idx]),
                    sample_model='ema')
                gen_sample.orig = EditDataSample(
                    fake_img=PixelData(data=outputs['orig'][idx]),
                    sample_model='orig')
                gen_sample.sample_model = 'ema/orig'
                gen_sample.set_gt_label(labels[idx])
                gen_sample.ema.set_gt_label(labels[idx])
                gen_sample.orig.set_gt_label(labels[idx])
            else:
                gen_sample.fake_img = PixelData(data=outputs[idx])
                gen_sample.set_gt_label(labels[idx])

            # Append input condition (noise and sample_kwargs) to
            # batch_sample_list
            if init_image is not None:
                gen_sample.noise = init_image[idx]
            gen_sample.sample_kwargs = deepcopy(sample_kwargs)
            batch_sample_list.append(gen_sample)
        return batch_sample_list

    def val_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data.

        Calls ``self.data_preprocessor(data)`` and
        ``self(inputs, data_sample, mode=None)`` in order. Return the
        generated results which will be passed to evaluator.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            SampleList: Generated image or image dict.
        """
        data = self.data_preprocessor(data)
        outputs = self(**data)
        return outputs

    def test_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            List[EditDataSample]: Generated image or image dict.
        """
        data = self.data_preprocessor(data)
        outputs = self(**data)
        return outputs
