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

from mmagic.registry import DIFFUSION_SCHEDULERS, MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import ForwardInputs, SampleList

ModelType = Union[Dict, nn.Module]


def classifier_grad(classifier, x, t, y=None, classifier_scale=1.0):
    """compute classification gradient to x."""
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale


@MODELS.register_module('GLIDE')
@MODELS.register_module()
class Glide(BaseModel):
    """GLIDE: Guided language to image diffusion for generation and editing.
        Refer to: https://github.com/openai/glide-text2im.


    Args:
        data_preprocessor (dict, optional): The pre-process configuration for
            :class:`BaseDataPreprocessor`.
        unet (ModelType): Configuration for the denoising Unet.
        diffusion_scheduler (ModelType): Configuration for the diffusion
            scheduler.
        unet_up (ModelType, optional): Configuration for the upsampling
            denoising UNet. Defaults to None.
        diffusion_scheduler_up (ModelType, optional): Configuration for
            the upsampling diffusion scheduler. Defaults to None.
        use_fp16 (bool, optional): Whether to use fp16 for the unet model.
            Defaults to False.
        classifier (ModelType, optional): Configuration for the classifier.
            Defaults to None.
        classifier_scale (float): Classifier scale for classifier guidance.
            Defaults to 1.0.
        data_preprocessor (Optional[ModelType]): Configuration for the data
            preprocessor.
        pretrained_cfgs (dict, optional): Path configuration for pretrained
            weights. Usually, this is a dict containing the module name and
            the corresponding ckpt path. Defaults to None.
    """

    def __init__(self,
                 unet: ModelType,
                 diffusion_scheduler: ModelType,
                 unet_up: Optional[ModelType] = None,
                 diffusion_scheduler_up: Optional[ModelType] = None,
                 use_fp16: Optional[bool] = False,
                 classifier: Optional[dict] = None,
                 classifier_scale: float = 1.0,
                 data_preprocessor: Optional[ModelType] = dict(
                     type='DataPreprocessor'),
                 pretrained_cfgs: Optional[dict] = None):

        super().__init__(data_preprocessor=data_preprocessor)
        self.unet = unet if isinstance(unet, nn.Module) else MODELS.build(unet)
        self.diffusion_scheduler = DIFFUSION_SCHEDULERS.build(
            diffusion_scheduler) if isinstance(diffusion_scheduler,
                                               dict) else diffusion_scheduler

        self.unet_up = None
        self.diffusion_scheduler_up = None
        if unet_up:
            self.unet_up = unet_up if isinstance(
                unet_up, nn.Module) else MODELS.build(unet_up)
            if diffusion_scheduler_up:
                self.diffusion_scheduler_up = DIFFUSION_SCHEDULERS.build(
                    diffusion_scheduler_up) if isinstance(
                        diffusion_scheduler_up,
                        dict) else diffusion_scheduler_up
            else:
                self.diffusion_scheduler_up = deepcopy(
                    self.diffusion_scheduler)

        if classifier:
            self.classifier = MODELS.build(classifier)
        else:
            self.classifier = None
        self.classifier_scale = classifier_scale

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

    @torch.no_grad()
    def infer(self,
              init_image: Optional[torch.Tensor] = None,
              prompt: str = None,
              batch_size: Optional[int] = 1,
              guidance_scale: float = 3.,
              num_inference_steps: int = 50,
              num_inference_steps_up: Optional[int] = 27,
              labels: Optional[torch.Tensor] = None,
              classifier_scale: float = 0.0,
              show_progress: Optional[bool] = False):
        """Inference function for guided diffusion.

        Args:
            init_image (torch.Tensor, optional): Starting noise for diffusion.
                Defaults to None.
            prompt (str): The prompt to guide the image generation.
            batch_size (int, optional): Batch size for generation.
                Defaults to 1.
            num_inference_steps (int, optional): The number of denoising steps.
                Defaults to 50.
            num_inference_steps_up (int, optional): The number of upsampling
                denoising steps. Defaults to 27.
            labels (torch.Tensor, optional): Labels for the classifier.
                Defaults to None.
            show_progress (bool, optional): Whether to show the progress bar.
                Defaults to False.

        Returns:
            torch.Tensor: Generated images.
        """
        # Sample gaussian noise to begin loop
        if init_image is None:
            image = torch.randn((2 * batch_size, self.unet.in_channels,
                                 self.unet.image_size, self.unet.image_size))
            image = image.to(self.device)
        else:
            image = init_image

        # set step values
        if num_inference_steps > 0:
            self.diffusion_scheduler.set_timesteps(num_inference_steps)

        timesteps = self.diffusion_scheduler.timesteps

        # text embedding
        tokens = self.unet.tokenizer.encode(prompt)
        tokens, mask = self.unet.tokenizer.padded_tokens_and_mask(tokens, 128)

        # Create the classifier-free guidance tokens (empty)
        # full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = \
            self.unet.tokenizer.padded_tokens_and_mask(
                [], 128)

        tokens = torch.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size,
            device=self.device)
        mask = torch.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=torch.bool,
            device=self.device)

        if show_progress and mmengine.dist.is_main_process():
            timesteps = tqdm(timesteps)

        for t in timesteps:
            # 1. predicted model_output
            half = image[:len(image) // 2]
            combined = torch.cat([half, half], dim=0)
            model_output = self.unet(combined, t, tokens=tokens, mask=mask)
            eps, rest = model_output[:, :3], model_output[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            noise_pred = torch.cat([eps, rest], dim=1)

            # 2. compute previous image: x_t -> x_t-1
            diffusion_scheduler_output = self.diffusion_scheduler.step(
                noise_pred, t, image)

            # 3. applying classifier guide
            if self.classifier and classifier_scale != 0.0:
                gradient = classifier_grad(
                    self.classifier,
                    image,
                    t,
                    labels,
                    classifier_scale=classifier_scale)
                guided_mean = (
                    diffusion_scheduler_output['mean'].float() +
                    diffusion_scheduler_output['sigma'] * gradient.float())
                image = guided_mean + diffusion_scheduler_output[
                    'sigma'] * diffusion_scheduler_output['noise']
            else:
                image = diffusion_scheduler_output['prev_sample']

        # abandon unconditional image
        image = image[:image.shape[0] // 2]

        if self.unet_up:
            image = self.infer_up(
                low_res_img=image,
                batch_size=batch_size,
                prompt=prompt,
                num_inference_steps=num_inference_steps_up)

        return {'samples': image}

    @torch.no_grad()
    def infer_up(self,
                 low_res_img: torch.Tensor,
                 batch_size: int = 1,
                 init_image: Optional[torch.Tensor] = None,
                 prompt: Optional[str] = None,
                 num_inference_steps: int = 27,
                 show_progress: bool = False):
        """Inference function for upsampling guided diffusion.

        Args:
            low_res_img (torch.Tensor): Low resolution image
                (shape: [B, C, H, W]) for upsampling.
            batch_size (int, optional): Batch size for generation.
                Defaults to 1.
            init_image (torch.Tensor, optional): Starting noise
                (shape: [B, C, H, W]) for diffusion. Defaults to None.
            prompt (str, optional): The text prompt to guide the image
                generation. Defaults to None.
            num_inference_steps (int, optional): The number of denoising
                steps. Defaults to 27.
            show_progress (bool, optional): Whether to show the progress bar.
                Defaults to False.

        Returns:
            torch.Tensor: Generated upsampled images (shape: [B, C, H, W]).
        """
        if init_image is None:
            image = torch.randn(
                (batch_size, self.unet_up.in_channels // 2,
                 self.unet_up.image_size, self.unet_up.image_size))
            image = image.to(self.device)
        else:
            image = init_image

        # set step values
        if num_inference_steps > 0:
            self.diffusion_scheduler_up.set_timesteps(num_inference_steps)
        timesteps = self.diffusion_scheduler_up.timesteps

        # text embedding
        tokens = self.unet.tokenizer.encode(prompt)
        tokens, mask = self.unet.tokenizer.padded_tokens_and_mask(tokens, 128)
        tokens = torch.tensor(
            [tokens] * batch_size, dtype=torch.bool, device=self.device)
        mask = torch.tensor(
            [mask] * batch_size, dtype=torch.bool, device=self.device)

        if show_progress and mmengine.dist.is_main_process():
            timesteps = tqdm(timesteps)

        for t in timesteps:
            noise_pred = self.unet_up(
                image, t, low_res=low_res_img, tokens=tokens, mask=mask)
            # compute previous image: x_t -> x_t-1
            diffusion_scheduler_output = self.diffusion_scheduler_up.step(
                noise_pred, t, image)
            image = diffusion_scheduler_output['prev_sample']

        return image

    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> List[DataSample]:
        """_summary_

        Args:
            inputs (ForwardInputs): _description_
            data_samples (Optional[list], optional): _description_.
                Defaults to None.
            mode (Optional[str], optional): _description_. Defaults to None.

        Returns:
            List[DataSample]: _description_
        """
        init_image = inputs.get('init_image', None)
        batch_size = inputs.get('batch_size', 1)
        labels = data_samples.get('labels', None)
        sample_kwargs = inputs.get('sample_kwargs', dict())

        num_inference_steps = sample_kwargs.get(
            'num_inference_steps',
            self.diffusion_scheduler.num_train_timesteps)
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
            gen_sample = DataSample()
            if data_samples:
                gen_sample.update(data_samples[idx])
            if isinstance(outputs, dict):
                gen_sample.ema = DataSample(
                    fake_img=outputs['ema'][idx], sample_model='ema')
                gen_sample.orig = DataSample(
                    fake_img=outputs['orig'][idx], sample_model='orig')
                gen_sample.sample_model = 'ema/orig'
                gen_sample.set_gt_label(labels[idx])
                gen_sample.ema.set_gt_label(labels[idx])
                gen_sample.orig.set_gt_label(labels[idx])
            else:
                gen_sample.fake_img = outputs[idx]
                gen_sample.set_gt_label(labels[idx])

            # Append input condition (noise and sample_kwargs) to
            # batch_sample_list
            if init_image is not None:
                gen_sample.noise = init_image[idx]
            gen_sample.sample_kwargs = deepcopy(sample_kwargs)
            batch_sample_list.append(gen_sample)
        return batch_sample_list

    @torch.no_grad()
    def val_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data.

        Calls ``self.data_preprocessor(data)`` and
        ``self(inputs, data_sample, mode=None)`` in order. Return the
        generated results which will be passed to evaluator.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More details in `Metrics` and `Evaluator`.

        Returns:
            SampleList: Generated image or image dict.
        """
        data = self.data_preprocessor(data)
        outputs = self(**data)
        return outputs

    @torch.no_grad()
    def test_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More details in `Metrics` and `Evaluator`.

        Returns:
            List[DataSample]: Generated image or image dict.
        """
        data = self.data_preprocessor(data)
        outputs = self(**data)
        return outputs

    def train_step(self, data: dict, optim_wrapper: OptimWrapperDict):
        """_summary_

        Args:
            data (dict): _description_
            optim_wrapper (OptimWrapperDict): _description_

        Returns:
            _type_: _description_
        """
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')

        # sampling x0 and timestep
        data = self.data_preprocessor(data)
        real_imgs = data['inputs']
        timestep = self.diffusion_scheduler.sample_timestep()

        # calculating loss
        loss_dict = self.diffusion_scheduler.training_loss(
            self.unet, real_imgs, timestep)
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
