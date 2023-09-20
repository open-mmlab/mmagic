# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import random
from datetime import datetime
from glob import glob

import gradio as gr
import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from mmengine import Config
from safetensors import safe_open

from mmagic.models.editors.animatediff import save_videos_grid
from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

sample_idx = 0
scheduler_dict = {
    'Euler': EulerDiscreteScheduler,
    'PNDM': PNDMScheduler,
    'DDIM': DDIMScheduler,
}
cfg = Config.fromfile('configs/animatediff/animatediff_ToonYou.py')
css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""


class AnimateController:

    def __init__(self):

        # config dirs
        self.basedir = cfg.models_path
        self.stable_diffusion_dir = os.path.join(self.basedir,
                                                 'StableDiffusion')
        self.motion_module_dir = os.path.join(self.basedir, 'Motion_Module')
        self.personalized_model_dir = os.path.join(self.basedir,
                                                   'DreamBooth_LoRA')
        self.savedir = os.path.join(
            os.getcwd(), 'samples',
            datetime.now().strftime('Gradio-%Y-%m-%dT%H-%M-%S'))
        self.savedir_sample = os.path.join(self.savedir, 'sample')
        os.makedirs(self.savedir, exist_ok=True)

        self.stable_diffusion_list = []
        self.motion_module_list = []
        self.personalized_model_list = []

        self.refresh_stable_diffusion()
        self.refresh_motion_module()
        self.refresh_personalized_model()

        # config models
        self.config = cfg
        self.animatediff = None
        self.lora_model_state_dict = {}

    def refresh_stable_diffusion(self):
        self.stable_diffusion_list = ['runwayml/stable-diffusion-v1-5']

    def refresh_motion_module(self):
        motion_module_list = glob(
            os.path.join(self.motion_module_dir, '*.ckpt'))
        self.motion_module_list = [
            os.path.basename(p) for p in motion_module_list
        ]

    def refresh_personalized_model(self):
        personalized_model_list = glob(
            os.path.join(self.personalized_model_dir, '*.safetensors'))
        self.personalized_model_list = [
            os.path.basename(p) for p in personalized_model_list
        ]

    def update_stable_diffusion(self, stable_diffusion_dropdown):
        self.config['stable_diffusion_v15_url'] = stable_diffusion_dropdown
        self.animatediff = MODELS.build(self.config.model).cuda()
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        if self.animatediff.unet is None:
            gr.Info('Please select a pretrained model path.')
            return gr.Dropdown.update(value=None)
        else:
            motion_module_dropdown = os.path.join(self.motion_module_dir,
                                                  motion_module_dropdown)
            self.config.model['motion_module_cfg']['path'] = \
                motion_module_dropdown
            self.animatediff.init_motion_module(
                self.config.model['motion_module_cfg'])
            return gr.Dropdown.update()

    def update_base_model(self, base_model_dropdown):

        if self.animatediff.unet is None:
            gr.Info('Please select a pretrained model path.')
            return gr.Dropdown.update(value=None)
        else:
            base_model_dropdown = os.path.join(self.personalized_model_dir,
                                               base_model_dropdown)
            self.config.model['dream_booth_lora_cfg'][
                'path'] = base_model_dropdown
            self.animatediff.init_dreambooth_lora(
                self.config.model['dream_booth_lora_cfg'])
            return gr.Dropdown.update()

    def update_lora_model(self, lora_model_dropdown):
        lora_model_dropdown = os.path.join(self.personalized_model_dir,
                                           lora_model_dropdown)
        self.lora_model_state_dict = {}
        if lora_model_dropdown == 'none':
            pass
        else:
            with safe_open(
                    lora_model_dropdown, framework='pt', device='cpu') as f:
                for key in f.keys():
                    self.lora_model_state_dict[key] = f.get_tensor(key)
        return gr.Dropdown.update()

    def animate(
            self,
            stable_diffusion_dropdown,
            motion_module_dropdown,
            base_model_dropdown,
            lora_alpha_slider,
            prompt_textbox,
            negative_prompt_textbox,
            # sampler_dropdown,
            sample_step_slider,
            width_slider,
            length_slider,
            height_slider,
            cfg_scale_slider,
            seed_textbox):
        if self.animatediff.unet is None:
            raise gr.Error('Please select a pretrained model path.')
        if motion_module_dropdown == '':
            raise gr.Error('Please select a motion module.')
        if base_model_dropdown == '':
            raise gr.Error('Please select a base DreamBooth model.')
        self.animatediff.cuda()
        self.animatediff.unet.set_use_memory_efficient_attention_xformers(True)

        # TODO: update lora
        # if self.lora_model_state_dict != {}:
        #     pipeline = convert_lora(
        # pipeline,
        # self.lora_model_state_dict,
        # alpha=lora_alpha_slider)

        sample = self.animatediff.infer(
            prompt_textbox,
            negative_prompt=negative_prompt_textbox,
            num_inference_steps=sample_step_slider,
            guidance_scale=cfg_scale_slider,
            width=width_slider,
            height=height_slider,
            video_length=length_slider,
            seed=int(seed_textbox))['samples']

        save_sample_path = os.path.join(self.savedir_sample,
                                        f'{sample_idx}.mp4')
        save_videos_grid(sample, save_sample_path)
        seed = torch.initial_seed()
        sample_config = {
            'prompt': prompt_textbox,
            'n_prompt': negative_prompt_textbox,
            # "sampler": sampler_dropdown, # TODO: More samplers
            'num_inference_steps': sample_step_slider,
            'guidance_scale': cfg_scale_slider,
            'width': width_slider,
            'height': height_slider,
            'video_length': length_slider,
            'seed': seed
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, 'logs.json'), 'a') as f:
            f.write(json_str)
            f.write('\n\n')

        return gr.Video.update(value=save_sample_path)


controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("""
            # [AnimateDiff: Animate Your Personalized
            # Text-to-Image Diffusion Models without Specific Tuning]
            # (https://arxiv.org/abs/2307.04725)
            Yuwei Guo, Ceyuan Yang*, Anyi Rao, Yaohui Wang,
            Yu Qiao, Dahua Lin, Bo Dai (*Corresponding Author)<br>
            [Arxiv Report](https://arxiv.org/abs/2307.04725) |
            [Project Page](https://animatediff.github.io/) |
            [Github](https://github.com/guoyww/animatediff/)
            """)
        with gr.Column(variant='panel'):
            gr.Markdown("""
                ### 1. Model checkpoints (select pretrained model path first).
                """)
            with gr.Row():
                stable_diffusion_dropdown = gr.Dropdown(
                    label='Pretrained Model Path',
                    choices=controller.stable_diffusion_list,
                    interactive=True,
                )
                stable_diffusion_dropdown.change(
                    fn=controller.update_stable_diffusion,
                    inputs=[stable_diffusion_dropdown],
                    outputs=[stable_diffusion_dropdown])

                stable_diffusion_refresh_button = gr.Button(
                    value='\U0001F503', elem_classes='toolbutton')

                def update_stable_diffusion():
                    controller.refresh_stable_diffusion()
                    return gr.Dropdown.update(
                        choices=controller.stable_diffusion_list)

                stable_diffusion_refresh_button.click(
                    fn=update_stable_diffusion,
                    inputs=[],
                    outputs=[stable_diffusion_dropdown])

            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label='Select motion module',
                    choices=controller.motion_module_list,
                    interactive=True,
                )
                motion_module_dropdown.change(
                    fn=controller.update_motion_module,
                    inputs=[motion_module_dropdown],
                    outputs=[motion_module_dropdown])

                motion_module_refresh_button = gr.Button(
                    value='\U0001F503', elem_classes='toolbutton')

                def update_motion_module():
                    controller.refresh_motion_module()
                    return gr.Dropdown.update(
                        choices=controller.motion_module_list)

                motion_module_refresh_button.click(
                    fn=update_motion_module,
                    inputs=[],
                    outputs=[motion_module_dropdown])

                base_model_dropdown = gr.Dropdown(
                    label='Select base Dreambooth model (required)',
                    choices=controller.personalized_model_list,
                    interactive=True,
                )
                base_model_dropdown.change(
                    fn=controller.update_base_model,
                    inputs=[base_model_dropdown],
                    outputs=[base_model_dropdown])

                lora_model_dropdown = gr.Dropdown(
                    label='Select LoRA model (optional)',
                    choices=['none'] + controller.personalized_model_list,
                    value='none',
                    interactive=True,
                )
                lora_model_dropdown.change(
                    fn=controller.update_lora_model,
                    inputs=[lora_model_dropdown],
                    outputs=[lora_model_dropdown])

                lora_alpha_slider = gr.Slider(
                    label='LoRA alpha',
                    value=0.8,
                    minimum=0,
                    maximum=2,
                    interactive=True)

                personalized_refresh_button = gr.Button(
                    value='\U0001F503', elem_classes='toolbutton')

                def update_personalized_model():
                    controller.refresh_personalized_model()
                    return [
                        gr.Dropdown.update(
                            choices=controller.personalized_model_list),
                        gr.Dropdown.update(choices=['none'] +
                                           controller.personalized_model_list)
                    ]

                personalized_refresh_button.click(
                    fn=update_personalized_model,
                    inputs=[],
                    outputs=[base_model_dropdown, lora_model_dropdown])

        with gr.Column(variant='panel'):
            gr.Markdown("""
                ### 2. Configs for AnimateDiff.
                """)

            prompt_textbox = gr.Textbox(label='Prompt', lines=2)
            negative_prompt_textbox = gr.Textbox(
                label='Negative prompt', lines=2)

            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        sample_step_slider = gr.Slider(
                            label='Sampling steps',
                            value=25,
                            minimum=10,
                            maximum=100,
                            step=1)

                    width_slider = gr.Slider(
                        label='Width',
                        value=512,
                        minimum=256,
                        maximum=1024,
                        step=64)
                    height_slider = gr.Slider(
                        label='Height',
                        value=512,
                        minimum=256,
                        maximum=1024,
                        step=64)
                    length_slider = gr.Slider(
                        label='Animation length',
                        value=16,
                        minimum=8,
                        maximum=24,
                        step=1)
                    cfg_scale_slider = gr.Slider(
                        label='CFG Scale', value=7.5, minimum=0, maximum=20)

                    with gr.Row():
                        seed_textbox = gr.Textbox(label='Seed', value=-1)
                        seed_button = gr.Button(
                            value='\U0001F3B2', elem_classes='toolbutton')
                        seed_button.click(
                            fn=lambda: gr.Textbox.update(
                                value=random.randint(1, 1e8)),
                            inputs=[],
                            outputs=[seed_textbox])

                    generate_button = gr.Button(
                        value='Generate', variant='primary')

                result_video = gr.Video(
                    label='Generated Animation', interactive=False)

            generate_button.click(
                fn=controller.animate,
                inputs=[
                    stable_diffusion_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    lora_alpha_slider,
                    prompt_textbox,
                    negative_prompt_textbox,
                    # sampler_dropdown,
                    sample_step_slider,
                    width_slider,
                    length_slider,
                    height_slider,
                    cfg_scale_slider,
                    seed_textbox,
                ],
                outputs=[result_video])

    return demo


if __name__ == '__main__':
    demo = ui()
    demo.launch(share=True)
