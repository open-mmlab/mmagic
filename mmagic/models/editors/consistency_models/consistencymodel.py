# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine import Config
from mmengine.model import BaseModel
from torch.hub import load_state_dict_from_url

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import ForwardInputs
from .consistencymodel_utils import (device, get_generator, get_sample_fn,
                                     get_sigmas_karras, karras_sample)

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class ConsistencyModel(BaseModel, metaclass=ABCMeta):
    """Implementation of `ConsistencyModel.

    <https://arxiv.org/abs/2303.01469>`_ (ConsistencyModel).
    """

    def __init__(self,
                 unet: ModelType,
                 denoiser: ModelType,
                 attention_resolutions: str = '32,16,8',
                 batch_size: int = 4,
                 channel_mult: str = '',
                 class_cond: Union[bool, int] = False,
                 generator: str = 'determ-indiv',
                 image_size: int = 256,
                 learn_sigma: bool = False,
                 model_path: Optional[str] = None,
                 num_classes: int = 0,
                 num_samples: int = 0,
                 sampler: str = 'onestep',
                 seed: int = 0,
                 sigma_max: float = 80.0,
                 sigma_min: float = 0.002,
                 training_mode: str = 'consistency_distillation',
                 ts: str = '',
                 clip_denoised: bool = True,
                 s_churn: float = 0.0,
                 s_noise: float = 1.0,
                 s_tmax: float = float('inf'),
                 s_tmin: float = 0.0,
                 steps: int = 40,
                 data_preprocessor: Optional[Union[dict, Config]] = None):

        super().__init__(data_preprocessor=data_preprocessor)

        self.num_classes = num_classes
        if 'consistency' in training_mode:
            self.distillation = True
        else:
            self.distillation = False
        self.batch_size = batch_size
        self.class_cond = class_cond
        self.image_size = image_size
        self.sampler = sampler
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.clip_denoised = clip_denoised
        self.s_churn = s_churn
        self.s_noise = s_noise
        self.s_tmax = s_tmax
        self.s_tmin = s_tmin
        self.steps = steps
        self.model_kwargs = {}

        if channel_mult == '':
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            else:
                raise ValueError(f'unsupported image size: {image_size}')
        else:
            channel_mult = tuple(
                int(ch_mult) for ch_mult in channel_mult.split(','))

        attention_ds = []
        for res in attention_resolutions.split(','):
            attention_ds.append(image_size // int(res))

        if isinstance(unet, dict):
            unet['image_size'] = image_size
            unet['out_channels'] = (3 if not learn_sigma else 6)
            unet['num_classes'] = (num_classes if class_cond else None)
            unet['attention_resolutions'] = tuple(attention_ds)
            unet['channel_mult'] = channel_mult
            self.model = MODELS.build(unet)
        else:
            self.model = unet

        if isinstance(denoiser, dict):
            denoiser['distillation'] = self.distillation
            self.diffusion = MODELS.build(denoiser)
        else:
            self.diffusion = denoiser

        if model_path:
            if 'https://' in model_path or 'http://' in model_path:
                self.model.load_state_dict(
                    load_state_dict_from_url(model_path, map_location='cpu'))
            else:
                self.model.load_state_dict(
                    torch.load(model_path, map_location='cpu'))

        self.model.to(device())

        if sampler == 'multistep':
            assert len(ts) > 0
            self.ts = tuple(int(x) for x in ts.split(','))
        else:
            self.ts = None

        self.all_images = []
        self.all_labels = []
        if num_samples <= 0:
            self.num_samples = batch_size
        else:
            self.num_samples = num_samples
        self.generator = get_generator(generator, self.num_samples, seed)

    def infer(self, class_id: Optional[int] = None):
        """infer with unet model and diffusion."""
        self.model.eval()
        while len(self.all_images) * self.batch_size < self.num_samples:
            self.model_kwargs = {}
            if self.class_cond:
                classes = self.label_fn(class_id)
                self.model_kwargs['y'] = classes
            sample = karras_sample(
                self.diffusion,
                self.model,
                (self.batch_size, 3, self.image_size, self.image_size),
                steps=self.steps,
                model_kwargs=self.model_kwargs,
                device=device(),
                clip_denoised=self.clip_denoised,
                sampler=self.sampler,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                s_churn=self.s_churn,
                s_tmin=self.s_tmin,
                s_tmax=self.s_tmax,
                s_noise=self.s_noise,
                generator=self.generator,
                ts=self.ts,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            self.all_images.extend([sample.cpu().numpy()])
            if self.class_cond:
                self.all_labels.extend([classes.cpu().numpy()])

        arr = np.concatenate(self.all_images, axis=0)
        arr = arr[:self.num_samples]
        label_arr = []
        if self.class_cond:
            label_arr = np.concatenate(self.all_labels, axis=0)
            label_arr = label_arr[:self.num_samples]

        return arr, label_arr

    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> List[DataSample]:
        """Sample images with the given inputs. If forward mode is 'ema' or
        'orig', the image generated by corresponding generator will be
        returned. If forward mode is 'ema/orig', images generated by original
        generator and EMA generator will both be returned in a dict.

        Args:
            inputs (ForwardInputs): Dict containing the necessary
                information (e.g. noise, num_batches, mode) to generate image.
            data_samples (Optional[list]): Data samples collated by
                :attr:`data_preprocessor`. Defaults to None.
            mode (Optional[str]): `mode` is not used in
                :class:`BaseConditionalGAN`. Defaults to None.

        Returns:
            List[DataSample]: Generated images or image dict.
        """

        self.model_kwargs = {}
        progress = False
        callback = None
        sample_kwargs = inputs.get('sample_kwargs', dict())
        labels = self.label_fn(inputs.get('labels', 'None'))
        if self.class_cond:
            assert len(labels) > 0, 'If class_cond is True, ' \
                                    'labels\'s size should be over zero.'
            self.model_kwargs['y'] = labels
        sample_model = inputs.get('sample_model', None)
        if self.generator is None:
            self.generator = get_generator('dummy')

        if self.sampler == 'progdist':
            sigmas = get_sigmas_karras(
                self.steps + 1,
                self.sigma_min,
                self.sigma_max,
                self.diffusion.rho,
                device=device())
        else:
            sigmas = get_sigmas_karras(
                self.steps,
                self.sigma_min,
                self.sigma_max,
                self.diffusion.rho,
                device=device())

        noise = self.generator.randn(
            *(self.batch_size, 3, self.image_size, self.image_size),
            device=device()) * self.sigma_max

        sample_fn = get_sample_fn(self.sampler)

        if self.sampler in ['heun', 'dpm']:
            sampler_args = dict(
                s_churn=self.s_churn,
                s_tmin=self.s_tmin,
                s_tmax=self.s_tmax,
                s_noise=self.s_noise)
        elif self.sampler == 'multistep':
            sampler_args = dict(
                ts=self.ts,
                t_min=self.sigma_min,
                t_max=self.sigma_max,
                rho=self.diffusion.rho,
                steps=self.steps)
        else:
            sampler_args = {}

        outputs = sample_fn(
            self.denoiser,
            noise,
            sigmas,
            self.generator,
            progress=progress,
            callback=callback,
            **sampler_args,
        ).clamp(-1, 1)
        outputs = self.data_preprocessor.destruct(outputs, data_samples)
        outputs = self.data_preprocessor._do_conversion(outputs, 'BGR',
                                                        'RGB')[0]

        gen_sample = DataSample()
        if data_samples:
            gen_sample.update(data_samples)
        gen_sample.fake_img = outputs
        gen_sample.noise = noise
        gen_sample.set_gt_label(labels)
        gen_sample.sample_kwargs = deepcopy(sample_kwargs)
        gen_sample.sample_model = sample_model
        batch_sample_list = gen_sample.split(allow_nonseq_value=True)

        return batch_sample_list

    def denoiser(self, x_t, sigma):
        """return diffusion's denoiser."""
        _, denoised = self.diffusion.denoise(self.model, x_t, sigma,
                                             **self.model_kwargs)
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    def label_fn(self, class_id):
        """return random class_id if class_id is none."""
        assert self.num_classes > 0, \
            'If class_cond is not False,' \
            'num_classes should be larger than zero.'
        if class_id:
            assert -1 < int(class_id) < self.num_classes, \
                'If class_cond has been defined as a class_label_id, ' \
                'it should be within the range (0,num_classes).'
            classes = torch.tensor(
                [int(class_id) for i in range(self.batch_size)],
                device=device())
        else:
            classes = torch.randint(
                low=0,
                high=self.num_classes,
                size=(self.batch_size, ),
                device=device())
        return classes
