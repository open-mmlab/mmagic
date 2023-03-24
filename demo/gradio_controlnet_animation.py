# Copyright (c) OpenMMLab. All rights reserved.
import gradio as gr

from mmedit.edit import MMEdit


def process(video, prompt, a_prompt, negative_prompt,
            controlnet_conditioning_scale):

    editor = MMEdit(model_name='controlnet_animation')

    prompt = prompt + a_prompt

    save_path = editor.infer(
        video=video,
        prompt=prompt,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=0.7)

    return save_path


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown('## Controlnet Animation')
    with gr.Row():
        with gr.Column():
            video_inp = gr.Video(
                label='Video source',
                source='upload',
                type='filepath',
                elem_id='input-vid')
            prompt = gr.Textbox(label='Prompt')
            run_button = gr.Button(label='Run')
            with gr.Accordion('Advanced options', open=False):
                image_resolution = gr.Slider(
                    label='Image Resolution',
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64)
                a_prompt = gr.Textbox(
                    label='Added Prompt',
                    value='best quality, extremely detailed')
                n_prompt = gr.Textbox(
                    label='Negative Prompt',
                    value='longbody, lowres, bad anatomy, bad hands, ' +
                    'missing fingers, extra digit, fewer digits, '
                    'cropped, worst quality, low quality')
        with gr.Column():
            video_out = gr.Video(
                label='ControlNet video result', elem_id='video-output')
    ips = [video_inp, prompt, a_prompt, n_prompt, image_resolution]
    run_button.click(fn=process, inputs=ips, outputs=video_out)

block.launch(server_name='0.0.0.0', share=True)
