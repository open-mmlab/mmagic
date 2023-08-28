# Copyright (c) OpenMMLab. All rights reserved.
import os

import gradio as gr
import torch
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

checkpoint_dir = 'ckpts'
ckpts = os.listdir(checkpoint_dir)
cfg = Config.fromfile('configs/vico/batman.py')
vico = MODELS.build(cfg.model)


class DummyModel:

    def __init__(self, model):
        self.model = model


dummy_agent = DummyModel(vico)


def process(checkpoint, img_ref, prompt, negative_prompt, guidance_scale,
            width, height, seed, inference_steps, bs):
    state_dict = torch.load(os.path.join(checkpoint_dir, checkpoint))
    dummy_agent.model.load_state_dict(state_dict, strict=False)
    dummy_agent.model = dummy_agent.model.cuda()
    img_ref = Image.open(img_ref)
    with torch.no_grad():
        output = dummy_agent.model.infer(
            prompt=prompt,
            image_reference=img_ref,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=inference_steps,
            num_images_per_prompt=bs,
            seed=int(seed))['samples']

    return output


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown('## ViCo')
    with gr.Row():
        with gr.Column():
            checkpoint = gr.Dropdown(ckpts)
            img_ref = gr.Image(
                label='Image Reference',
                source='upload',
                type='filepath',
                elem_id='input-vid')
            prompt = gr.Textbox(label='Prompt')
            with gr.Accordion('Advanced options', open=False):
                n_prompt = gr.Textbox(label='Negative Prompt', value='')
                guidance_scale = gr.Slider(
                    label='CFG',
                    minimum=0.0,
                    maximum=13.0,
                    value=7.5,
                    step=0.5)
                width = gr.Slider(
                    label='Image Width',
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64)
                height = gr.Slider(
                    label='Image Width',
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64)
                seed = gr.Number(
                    label='Seed',
                    value=0,
                )
                num_inference_steps = gr.Slider(
                    label='Inference Steps',
                    minimum=20,
                    maximum=80,
                    value=50,
                    step=5)
                bs = gr.Slider(
                    label='Batch Size', minimum=1, maximum=4, value=1, step=1)

        with gr.Column():
            output = gr.Gallery(label='Output Image', elem_id='image-output')
            run_button = gr.Button(label='Run')

    ips = [
        checkpoint, img_ref, prompt, n_prompt, guidance_scale, width, height,
        seed, num_inference_steps, bs
    ]
    run_button.click(fn=process, inputs=ips, outputs=output)

if __name__ == '__main__':
    block.launch(server_name='0.0.0.0', share=True)
