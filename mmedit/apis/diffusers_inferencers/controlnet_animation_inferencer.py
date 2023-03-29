# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import PIL.Image
import torch
from controlnet_aux import HEDdetector
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionControlNetPipeline, StableDiffusionPipelineOutput)
from diffusers.training_utils import set_seed
from diffusers.utils import load_image, randn_tensor
from mmengine import mkdir_or_exist
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.infer import BaseInferencer
from mmengine.structures import BaseDataElement
from torchvision import utils

from mmedit.utils import ConfigType, SampleList

InputType = Union[str, int, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[BaseDataElement, SampleList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], BaseDataElement, List[BaseDataElement]]

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


class StableDiffusionControlNetPipelineImg2Img(
        StableDiffusionControlNetPipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image,
                     List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        latent_image: Union[torch.FloatTensor, PIL.Image.Image,
                            List[torch.FloatTensor],
                            List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor],
                                    None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: float = 1.0,
        strength: float = 1.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation.
                If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`,
                `List[torch.FloatTensor]` or `List[PIL.Image.Image]`):
                The ControlNet input condition. ControlNet uses this
                input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is
                passed to ControlNet as is. PIL.Image.Image` can
                also be accepted as an image. The control image is
                automatically resized to fit the output image.
            height (`int`, *optional*, defaults to
                self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to
                self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually
                lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale
                is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that
                are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
                If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined,
                one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper:
                https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or
                `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)]
                (https://pytorch.org/docs/stable/generated/
                torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian
                distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with
                different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied
                random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak
                text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt`
                input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily
                tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be
                generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/):
                `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.
                StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps`
                 steps during inference. The function will be
                called with the following arguments: `callback(step: int,
                timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called.
                  If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the
                  `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/
                diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float`,
                 *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by
                  `controlnet_conditioning_scale` before they are added
                to the residual in the original unet.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or
              `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if
              `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the
              generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image
              likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, image, height, width, callback_steps,
                          negative_prompt, prompt_embeds,
                          negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance
        # weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
        #  `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare image
        image = self.prepare_image(
            image,
            width,
            height,
            batch_size * num_images_per_prompt,
            num_images_per_prompt,
            device,
            self.controlnet.dtype,
        )

        latent_image = self.prepare_latent_image(latent_image,
                                                 self.controlnet.dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size *
                                               num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            latent_image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            noise=latents)

        # 7. Prepare extra step kwargs. TODO: Logic should
        #  ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=image,
                    return_dict=False,
                )

                down_block_res_samples = [
                    down_block_res_sample * controlnet_conditioning_scale
                    for down_block_res_sample in down_block_res_samples
                ]
                mid_block_res_sample *= controlnet_conditioning_scale

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or \
                        ((i + 1) > num_warmup_steps and
                         (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading,
        #  let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.unet.to('cpu')
            self.controlnet.to('cpu')
            torch.cuda.empty_cache()

        if output_type == 'latent':
            image = latents
            has_nsfw_concept = None
        elif output_type == 'pil':
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept)

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(
            int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self,
                        image,
                        timestep,
                        batch_size,
                        num_images_per_prompt,
                        dtype,
                        device,
                        generator=None,
                        noise=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f'`image` has to be of type `torch.Tensor`, '
                f' `PIL.Image.Image` or list but is {type(image)}')

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of '
                f' length {len(generator)}, but requested an effective batch'
                f' size of {batch_size}. Make sure the batch size '
                f' matches the length of the generators.')

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i:i +
                                      1]).latent_dist.sample(generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and \
                batch_size % init_latents.shape[0] == 0:
            raise ValueError(
                f'Cannot duplicate `image` of batch size'
                f' {init_latents.shape[0]} to {batch_size} text prompts.')
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        if noise is None:
            noise = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def prepare_latent_image(self, image, dtype):
        if isinstance(image, torch.Tensor):
            # Batch single image
            if image.ndim == 3:
                image = image.unsqueeze(0)

            image = image.to(dtype=dtype)
        else:
            # preprocess image
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0],
                                                      PIL.Image.Image):
                image = [np.array(i.convert('RGB'))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=dtype) / 127.5 - 1.0

        return image


class ControlnetAnimationInferencer(BaseInferencer):
    """Base inferencer.

    Args:
        config (str or ConfigType): Model config or the path to it.
        ckpt (str, optional): Path to the checkpoint.
        device (str, optional): Device to run inference. If None, the best
            device will be automatically used.
        result_out_dir (str): Output directory of images. Defaults to ''.
    """

    func_kwargs = dict(
        preprocess=[],
        forward=[],
        visualize=['result_out_dir'],
        postprocess=['get_datasample'])
    func_order = dict(preprocess=0, forward=1, visualize=2, postprocess=3)

    extra_parameters = dict()

    def __init__(self,
                 config: Union[ConfigType, str],
                 device: Optional[str] = None,
                 extra_parameters: Optional[Dict] = None,
                 dtype=torch.float16,
                 **kwargs) -> None:
        cfg = Config.fromfile(config)
        self.hed = HEDdetector.from_pretrained(cfg.control_detector)
        self.controlnet = ControlNetModel.from_pretrained(
            cfg.controlnet_model, torch_dtype=dtype)
        self.pipe = StableDiffusionControlNetPipelineImg2Img.from_pretrained(
            cfg.stable_diffusion_model,
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=dtype).to('cuda')

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config)

    @torch.no_grad()
    def __call__(self,
                 prompt=None,
                 video=None,
                 negative_prompt=None,
                 controlnet_conditioning_scale=0.7,
                 image_width=512,
                 image_height=512,
                 save_path=None,
                 strength=0.75,
                 num_inference_steps=20,
                 seed=1,
                 **kwargs) -> Union[Dict, List[Dict]]:
        """Call the inferencer.

        Args:
            kwargs: Keyword arguments for the inferencer.

        Returns:
            Union[Dict, List[Dict]]: Results of inference pipeline.
        """
        if save_path is None:
            from datetime import datetime
            datestring = datetime.now().strftime('%y%m%d-%H%M%S')
            save_path = '/tmp/' + datestring + '.mp4'

        set_seed(seed)

        init_noise_shape = (1, 4, image_height // 8, image_width // 8)
        init_noise_all_frame = torch.randn(
            init_noise_shape, dtype=self.controlnet.dtype).cuda()

        init_noise_shape_cat = (1, 4, image_height // 8, image_width // 8 * 3)
        init_noise_all_frame_cat = torch.randn(
            init_noise_shape_cat, dtype=self.controlnet.dtype).cuda()

        # load the images
        input_file_extension = os.path.splitext(video)[1]
        from_video = True
        all_images = []
        if input_file_extension in VIDEO_EXTENSIONS:
            video_reader = mmcv.VideoReader(video)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_path, fourcc, video_reader.fps,
                                           (image_width, image_height))
            for frame in video_reader:
                all_images.append(np.flip(frame, axis=2))
        else:
            frame_files = os.listdir(video)
            frame_files = [os.path.join(video, f) for f in frame_files]
            frame_files.sort()
            for frame in frame_files:
                frame_extension = os.path.splitext(frame)[1]
                if frame_extension in IMAGE_EXTENSIONS:
                    all_images.append(frame)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            from_video = False

        # first result
        image = None
        if from_video:
            image = PIL.Image.fromarray(all_images[0])
        else:
            image = load_image(all_images[0])
        image = image.resize((image_width, image_height))
        hed_image = self.hed(image, image_resolution=image_width)

        result = self.pipe(
            image=hed_image,
            latent_image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            latents=init_noise_all_frame).images[0]

        first_result = result
        first_hed = hed_image
        last_result = result
        last_hed = hed_image

        for ind in range(len(all_images)):
            if from_video:
                image = PIL.Image.fromarray(all_images[ind])
            else:
                image = load_image(all_images[ind])
            image = image.resize((image_width, image_height))
            hed_image = self.hed(image, image_resolution=image_width)

            concat_img = PIL.Image.new('RGB', (image_width * 3, image_height))
            concat_img.paste(last_result, (0, 0))
            concat_img.paste(image, (image_width, 0))
            concat_img.paste(first_result, (image_width * 2, 0))

            concat_hed = PIL.Image.new('RGB', (image_width * 3, image_height),
                                       'black')
            concat_hed.paste(last_hed, (0, 0))
            concat_hed.paste(hed_image, (image_width, 0))
            concat_hed.paste(first_hed, (image_width * 2, 0))

            result = self.pipe(
                image=concat_hed,
                latent_image=concat_img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                latents=init_noise_all_frame_cat,
            ).images[0]
            result = result.crop(
                (image_width, 0, image_width * 2, image_height))

            last_result = result
            last_hed = hed_image

            if from_video:
                video_writer.write(np.flip(np.asarray(result), axis=2))
            else:
                frame_name = frame_files[ind].split('/')[-1]
                save_name = os.path.join(save_path, frame_name)
                result.save(save_name)

        if from_video:
            video_writer.release()

        return save_path

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        if 'test_dataloader' in cfg and \
            'dataset' in cfg.test_dataloader and \
                'pipeline' in cfg.test_dataloader.dataset:
            pipeline_cfg = cfg.test_dataloader.dataset.pipeline
            return Compose(pipeline_cfg)
        return None

    def postprocess(
        self,
        preds: PredType,
        imgs: Optional[List[np.ndarray]] = None,
        is_batch: bool = False,
        get_datasample: bool = False,
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            imgs (Optional[np.ndarray]): Visualized predictions.
            is_batch (bool): Whether the inputs are in a batch.
                Defaults to False.
            get_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.

        Returns:
            result (Dict): Inference results as a dict.
            imgs (torch.Tensor): Image result of inference as a tensor or
                tensor list.
        """
        results = preds
        if not get_datasample:
            results = []
            for pred in preds:
                result = self._pred2dict(pred)
                results.append(result)
        if not is_batch:
            results = results[0]
        return results, imgs

    def _pred2dict(self, pred_tensor: torch.Tensor) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            pred_tensor (torch.Tensor): The tensor to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        result['infer_results'] = pred_tensor
        return result

    def visualize(self,
                  inputs: list,
                  preds: Any,
                  show: bool = False,
                  result_out_dir: str = '',
                  **kwargs) -> List[np.ndarray]:
        """Visualize predictions.

        Customize your visualization by overriding this method. visualize
        should return visualization results, which could be np.ndarray or any
        other objects.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (Any): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            result_out_dir (str): Output directory of images. Defaults to ''.

        Returns:
            List[np.ndarray]: Visualization results.
        """
        results = (preds[:, [2, 1, 0]] + 1.) / 2.

        # save images
        if result_out_dir:
            mkdir_or_exist(os.path.dirname(result_out_dir))
            utils.save_image(results, result_out_dir)

        return results
