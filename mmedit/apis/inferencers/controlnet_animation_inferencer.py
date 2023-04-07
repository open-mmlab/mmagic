# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Union

import cv2
import mmcv
import numpy as np
import PIL.Image
import torch
from controlnet_aux import HEDdetector
from diffusers import UniPCMultistepScheduler
from diffusers.training_utils import set_seed
from diffusers.utils import load_image
from mmengine.config import Config

from mmedit.registry import MODELS
from mmedit.utils import ConfigType
from .base_mmedit_inferencer import BaseMMEditInferencer

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


class ControlnetAnimationInferencer(BaseMMEditInferencer):
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
                 dtype=torch.float32,
                 **kwargs) -> None:
        cfg = Config.fromfile(config)
        self.hed = HEDdetector.from_pretrained(cfg.control_detector)
        self.pipe = MODELS.build(cfg.model).cuda()
        self.pipe.test_scheduler = UniPCMultistepScheduler.from_config(
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

        latent_width = image_width // 8
        latent_height = image_height // 8

        init_noise_shape = (1, 4, latent_height, latent_width)
        init_noise_all_frame = torch.randn(
            init_noise_shape, dtype=self.pipe.controlnet.dtype).cuda()

        init_noise_shape_cat = (1, 4, latent_height, latent_width * 3)
        init_noise_all_frame_cat = torch.randn(
            init_noise_shape_cat, dtype=self.pipe.controlnet.dtype).cuda()

        latent_mask = torch.zeros(
            (1, 4, image_height // 8, image_width // 8 * 3))
        latent_mask[:, :, :,
                    image_width // 8 + 1:image_width // 8 * 2 - 1] = 1.0
        latent_mask = latent_mask.type(self.pipe.controlnet.dtype).cuda()

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

        result = self.pipe.infer(
            control=hed_image,
            latent_image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            latents=init_noise_all_frame)['samples'][0]

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

            result = self.pipe.infer(
                control=concat_hed,
                latent_image=concat_img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                latents=init_noise_all_frame_cat,
                latent_mask=latent_mask,
            )['samples'][0]
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
