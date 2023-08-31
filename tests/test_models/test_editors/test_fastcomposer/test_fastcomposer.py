# Copyright (c) OpenMMLab. All rights reserved.
import gc
from unittest import TestCase

import numpy as np
import torch
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

gc.collect()
torch.cuda.empty_cache()

register_all_modules()
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'
clip_vit_url = 'openai/clip-vit-large-patch14'
finetuned_model_path = ''

config = dict(
    type='FastComposer',
    vae=dict(type='AutoencoderKL', sample_size=64),
    unet=dict(
        sample_size=64,
        type='UNet2DConditionModel',
        down_block_types=('DownBlock2D', ),
        up_block_types=('UpBlock2D', ),
        block_out_channels=(32, ),
        cross_attention_dim=16,
    ),
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
        self.fastcomposer = MODELS.build(config)

    def test_infer(self):
        fastcomposer = self.fastcomposer

        def mock_encode_prompt(prompt, do_classifier_free_guidance,
                               num_images_per_prompt, *args, **kwargs):
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            batch_size *= num_images_per_prompt
            if do_classifier_free_guidance:
                batch_size *= 2
            return torch.randn(batch_size, 5, 16)  # 2 for cfg

        encode_prompt = fastcomposer._encode_prompt
        fastcomposer._encode_prompt = mock_encode_prompt

        prompt = 'A man img'
        negative_prompt = ''
        alpha_ = 0.75
        guidance_scale = 5
        num_steps = 1
        num_images = 1
        image = []
        seed = -1
        augmented_prompt_embeds = torch.randn(1, 5, 16)
        image.append(
            Image.fromarray(
                np.random.randint(0, 255, size=(64, 64, 3)).astype('uint8')))

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

        result = fastcomposer.infer(
            prompt,
            negative_prompt=negative_prompt,
            height=64,
            width=64,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
            alpha_=alpha_,
            reference_subject_images=image,
            augmented_prompt_embeds=augmented_prompt_embeds,
            output_type='tensor')
        fastcomposer._encode_prompt = encode_prompt
        assert result['samples'].shape == (num_images, 3, 64, 64)
