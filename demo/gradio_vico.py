# Copyright (c) OpenMMLab. All rights reserved.
import os

import gradio as gr
import torch
from mmengine import Config
from mmengine.runner import Runner
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

checkpoint_dir = 'ckpts'
ckpts = os.listdir(checkpoint_dir)


def load_base_model():
    global cfg, model
    cfg = Config.fromfile('configs/vico/vico.py')
    model = MODELS.build(cfg.model)


def train(train_data, init_token, placeholder):
    data_root = 'tmp'
    concept_dir = init_token
    image_dir = os.path.join(data_root, concept_dir)
    os.makedirs(image_dir, exist_ok=True)
    for i, data in enumerate(train_data):
        image = Image.open(data.name)
        image.save(os.path.join(image_dir, f'{i}.png'))
    global train_cfg
    train_cfg = Config.fromfile('configs/vico/vico.py')
    train_cfg.work_dir = os.path.join('./work_dirs', 'vico_gradio')
    train_cfg.dataset['data_root'] = data_root
    train_cfg.dataset['concept_dir'] = init_token
    train_cfg.dataset['placeholder'] = placeholder
    train_cfg.train_dataloader['dataset'] = train_cfg.dataset
    train_cfg.model['placeholder'] = placeholder
    train_cfg.model['initialize_token'] = init_token
    train_cfg.custom_hooks = None
    runner = Runner.from_cfg(train_cfg)
    runner.train()


class DummyModel:

    def __init__(self, model):
        self.model = model


def infer_fn(checkpoint, img_ref, prompt, negative_prompt, guidance_scale,
             width, height, seed, inference_steps, bs):
    state_dict = torch.load(os.path.join(checkpoint_dir, checkpoint))
    dummy_agent = DummyModel(model)
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


if __name__ == '__main__':
    load_base_model()
    block = gr.Blocks().queue()
    with block:
        with gr.Tab('Train'):
            with gr.Row():
                gr.Markdown('## ViCo')
            with gr.Row():
                with gr.Column():
                    train_data = gr.File(
                        label='Training Samples',
                        file_count='directory',
                        file_types=['image'],
                        interactive=True,
                    )
                with gr.Column():
                    init_token = gr.Textbox(label='Init token')
                    placeholder = gr.Textbox(label='Placeholder')
                    train_button = gr.Button(value='Start Training')
        train_button.click(
            fn=train, inputs=[train_data, init_token, placeholder])
        with gr.Tab('Inference'):
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
                        n_prompt = gr.Textbox(
                            label='Negative Prompt', value='')
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
                            label='Batch Size',
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1)

                with gr.Column():
                    output = gr.Gallery(
                        label='Output Image', elem_id='image-output')
                    run_button = gr.Button(label='Run')

            ips = [
                checkpoint, img_ref, prompt, n_prompt, guidance_scale, width,
                height, seed, num_inference_steps, bs
            ]
            run_button.click(fn=infer_fn, inputs=ips, outputs=output)
    block.launch(server_name='0.0.0.0', share=True)
