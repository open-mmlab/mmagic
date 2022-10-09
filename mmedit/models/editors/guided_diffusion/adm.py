# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, List, Optional, Union

import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapperDict
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from tqdm import tqdm

from mmedit.registry import DIFFUSERS, MODELS, MODULES
from mmedit.structures import EditDataSample, PixelData
from mmedit.utils.typing import ForwardInputs, SampleList

ModelType = Union[Dict, nn.Module]


@MODELS.register_module('ADM')
@MODELS.register_module('GuidedDiffusion')
@MODELS.register_module()
class AblatedDiffusionModel(BaseModel):
    """Ablated diffusion model mentioned in the paper Diffusion Models Beat
    GANs on Image Synthesis (https://arxiv.org/pdf/2105.05233.pdf).

    Args:
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        unet (Union[Dict, nn.Module]): The config or the model of denoising
            unet.
        diffuser (Dict): The config of the diffusion scheduler.
        use_fp16 (bool, optional): Whether to use fp16 parameters for blocks
            of unet. Defaults to False.
        classifier (Optional[Union[Dict, nn.Module]]): The config or the model
            of the classifier. Defaults to None.
        pretrained_cfgs (Optional[Dict]): The configs of the pretrained
            weights. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor,
                 unet,
                 diffuser,
                 use_fp16=False,
                 classifier=None,
                 classifier_scale=1.0,
                 pretrained_cfgs=None):

        super().__init__(data_preprocessor=data_preprocessor)
        self.unet = MODULES.build(unet)
        self.diffuser = DIFFUSERS.build(diffuser)
        if classifier:
            self.classifier = MODULES.build(unet)
        else:
            self.classifier = None
        self.classifier_scale = classifier_scale

        if pretrained_cfgs:
            self.load_pretrained_models(pretrained_cfgs)
        if use_fp16:
            mmengine.print_log('Convert unet modules to floatpoint16')
            self.unet.convert_to_fp16()

    def load_pretrained_models(self, pretrained_cfgs):
        """Loading pretrained weights as described in pretrained_cfgs.

        The ``pretrained_cfgs`` is a dict whose keys are module names,
        and values are corresponding checkpoint config. With this function,
        you can load modules from different models.

        Example:
            pretrained_cfgs = dict(
                unet=dict(ckpt_path='openai/512x512_diffusion.pt'),
                classifier=dict(ckpt_path="adm-u_8xb32_imagenet-512x512.pth",
                    prefix='classifier'),
            )

        Args:
            pretrained_cfgs (Optional[Dict]): The configs of the pretrained
                weights. Defaults to None.
        """
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
              classifier_scale=0.0,
              show_progress=False):
        """_summary_

        Args:
            init_image (Optional[torch.Tensor]): The initial image for
                diffusion process. Defaults to None.
            batch_size (int, optional): The batch size of image. Defaults to 1.
            num_inference_steps (int, optional): Denoising steps of inference.
                Defaults to 1000.
            labels (int, optional): The label of generated images. If not
                given, random labels will be used. Defaults to None.
            show_progress (bool, optional): Whether show progress bar of
                denoising process. Defaults to False.

        Returns:
            Dict: The result of inference.
        """
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
            # 1. predicted model_output
            model_output = self.unet(image, t, label=labels)['outputs']

            # 2. compute previous image: x_t -> x_t-1
            diffuser_output = self.diffuser.step(model_output, t, image)

            # 3. applying classifier guide
            if self.classifier and classifier_scale != 0.0:
                gradient = self.classifier_grad(
                    image, t, labels, classifier_scale=classifier_scale)
                guided_mean = (
                    diffuser_output['mean'].float() +
                    diffuser_output['sigma'] * gradient.float())
                image = guided_mean + diffuser_output[
                    'sigma'] * diffuser_output['noise']
            else:
                image = diffuser_output['prev_sample']

        return {'samples': image}

    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> List[EditDataSample]:
        """Sample images with the given inputs.

        Args:
            batch_inputs (ForwardInputs): Dict containing the necessary
                information (e.g. noise, num_batches, mode) to generate image.
            data_samples (Optional[list]): Data samples collated by
                :attr:`data_preprocessor`. Defaults to None.
            mode (Optional[str]): `mode` is not used in
                :class:`AblatedDiffusionModel`. Defaults to None.

        Returns:
            SampleList: A list of ``EditDataSample`` contain generated results.
        """
        init_image = inputs.get('init_image', None)
        batch_size = inputs.get('batch_size', 1)
        labels = data_samples.get('labels', None)
        sample_kwargs = inputs.get('sample_kwargs', dict())

        num_inference_steps = sample_kwargs.get(
            'num_inference_steps', self.diffuser.num_train_timesteps)
        show_progress = sample_kwargs.get('show_progress', False)
        classifier_scale = sample_kwargs.get('classifier_scale',
                                             self.classifier_scale)

        outputs = self.infer(
            init_image=init_image,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            show_progress=show_progress,
            classifier_scale=classifier_scale)

        batch_sample_list = []
        for idx in range(batch_size):
            gen_sample = EditDataSample()
            if data_samples:
                gen_sample.update(data_samples[idx])

            gen_sample.fake_img = PixelData(data=outputs['samples'][idx])
            gen_sample.set_gt_label(labels[idx])

            # Append input condition (noise and sample_kwargs) to
            # batch_sample_list
            if init_image is not None:
                gen_sample.init_image = init_image[idx]
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

    def train_step(self, data: dict, optim_wrapper: OptimWrapperDict):
        """Train GAN model. In the training of GAN models, generator and
        discriminator are updated alternatively. In MMEditing's design,
        `self.train_step` is called with data input. Therefore we always update
        discriminator, whose updating is relay on real data, and then determine
        if the generator needs to be updated based on the current number of
        iterations. More details about whether to update generator can be found
        in :meth:`should_gen_update`.

        Args:
            data (dict): Data sampled from dataloader.
            optim_wrapper (OptimWrapperDict): OptimWrapperDict instance
                contains OptimWrapper of generator and discriminator.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')

        # sampling x0 and timestep
        data = self.data_preprocessor(data)
        real_imgs = data['inputs']
        timestep = self.diffuser.sample_timestep()

        # calculating loss
        loss_dict = self.diffuser.training_loss(self.unet, real_imgs, timestep)
        loss, log_vars = self._parse_losses(loss_dict)
        optim_wrapper['denoising'].update_params(loss)

        # update EMA
        if self.with_ema_denoising and (curr_iter + 1) >= self.ema_start:
            self.denoising_ema.update_parameters(
                self.denoising_ema.
                module if is_model_wrapper(self.denoising) else self.denoising)
            # if not update buffer, copy buffer from orig model
            if not self.denoising_ema.update_buffers:
                self.denoising_ema.sync_buffers(
                    self.denoising.module
                    if is_model_wrapper(self.denoising) else self.denoising)
        elif self.with_ema_denoising:
            # before ema, copy weights from orig
            self.denoising_ema.sync_parameters(
                self.denoising.
                module if is_model_wrapper(self.denoising) else self.denoising)

        return log_vars

    def classifier_grad(self, x, t, y=None, classifier_scale=1.0):
        """Compute classifier guidance gradient at timestep t.

        Args:
            x (torch.Tensor): Denoising result as timestep t.
            t (int): Current timestep.
            y (torch.Tensor, optional): The labels tensor.
                Defaults to None.
            classifier_scale (float, optional): The magnitude of
                classifier guidance. Defaults to 1.0.

        Returns:
            torch.Tensor: Gradient of log probability to ``x_in`` multiplied
                by scale.
        """
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(),
                                       x_in)[0] * classifier_scale
