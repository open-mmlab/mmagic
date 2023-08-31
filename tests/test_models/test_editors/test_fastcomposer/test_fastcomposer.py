# Copyright (c) OpenMMLab. All rights reserved.
import gc
from unittest import TestCase

import numpy as np
import torch
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

gc.collect()
torch.cuda.empty_cache()

register_all_modules()
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'
clip_vit_url = 'openai/clip-vit-large-patch14'
finetuned_model_path = 'https://download.openxlab.org.cn/models/xiaomile/'\
                       'fastcomposer/weight/pytorch_model.bin'

config = dict(
    type='FastComposer',
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='vae'),
    unet=dict(
        type='UNet2DConditionModel',
        subfolder='unet',
        from_pretrained=stable_diffusion_v15_url),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    pretrained_cfg=dict(
        finetuned_model_path=finetuned_model_path,
        enable_xformers_memory_efficient_attention=None,
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        image_encoder_name_or_path=clip_vit_url,
        revision=None,
        non_ema_revision=None,
        object_localization=None,
        object_localization_weight=0.01,
        localization_layers=5,
        mask_loss=None,
        mask_loss_prob=0.5,
        object_localization_threshold=1.0,
        object_localization_normalize=None,
        no_object_augmentation=True,
        object_resolution=256),
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    dtype='fp32',
    data_preprocessor=dict(type='DataPreprocessor'))


class TestFastComposer(TestCase):

    def setUp(self):
        cfg_file = Config.fromfile(config)
        self.fastcomposer = MODELS.build(cfg_file.model)

    def test_infer(self):
        prompt = 'A man img and a man img sitting in a park'
        negative_prompt = '((((ugly)))), (((duplicate))), ((morbid)), ' \
                          '((mutilated)), [out of frame], extra fingers, ' \
                          'mutated hands, ((poorly drawn hands)), ((poorly ' \
                          'drawn face)), (((mutation))), (((deformed))), ' \
                          '((ugly)), blurry, ((bad anatomy)), (((bad ' \
                          'proportions))), ((extra limbs)), cloned face, ' \
                          '(((disfigured))). out of frame, ugly, extra limbs,'\
                          ' (bad anatomy), gross proportions, (malformed ' \
                          'limbs), ((missing arms)), ((missing legs)), ' \
                          '(((extra arms))), (((extra legs))), mutated hands,'\
                          ' (fused fingers), (too many fingers), ' \
                          '(((long neck)))'
        alpha_ = 0.75
        guidance_scale = 5
        num_steps = 1
        num_images = 4
        image = []
        seed = -1

        image.append(
            Image.fromarray(
                np.random.randint(0, 255, size=(512, 512, 3)).astype('uint8')))

        image.append(
            Image.fromarray(
                np.random.randint(0, 255, size=(512, 512, 3)).astype('uint8')))

        if len(image) == 0:
            raise Exception('You need to upload at least one image.')

        num_subject_in_text = (np.array(
            self.fastcomposer.special_tokenizer.encode(prompt)) ==
                               self.fastcomposer.image_token_id).sum()
        if num_subject_in_text != len(image):
            raise Exception(
                "Number of subjects in the text description doesn't match "
                'the number of reference images, #text subjects: '
                f'{num_subject_in_text} #reference image: {len(image)}', )

        if seed == -1:
            seed = np.random.randint(0, 1000000)

        generator = torch.manual_seed(seed)

        result = self.fastcomposer.infer(
            prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
            alpha_=alpha_,
            reference_subject_images=image,
            output_type='latent')

        assert result['samples'].shape == (num_images, 3, 512, 512)
