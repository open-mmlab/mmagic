# Copyright (c) OpenMMLab. All rights reserved.
import gradio as gr

from mmagic.apis import MMagicInferencer

editor = MMagicInferencer(model_name='controlnet_animation')


def process(video, prompt, a_prompt, negative_prompt,
            controlnet_conditioning_scale, width, height):
    prompt = prompt + a_prompt
    save_path = editor.infer(
        video=video,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        controlnet_conditioning_scale=controlnet_conditioning_scale)

    return save_path


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown('## Controlnet Animation')
    with gr.Row():
        with gr.Column():
            video_inp = gr.Video(
                label='Video Source',
                source='upload',
                type='filepath',
                elem_id='input-vid')
            prompt = gr.Textbox(label='Prompt')
            run_button = gr.Button(label='Run')
            with gr.Accordion('Advanced options', open=False):
                a_prompt = gr.Textbox(
                    label='Added Prompt',
                    value='best quality, extremely detailed')
                n_prompt = gr.Textbox(
                    label='Negative Prompt',
                    value='longbody, lowres, bad anatomy, bad hands, ' +
                    'missing fingers, extra digit, fewer digits, '
                    'cropped, worst quality, low quality')
                controlnet_conditioning_scale = gr.Slider(
                    label='Control Weight',
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.01)
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

        with gr.Column():
            video_out = gr.Video(label='Video Result', elem_id='video-output')
    ips = [
        video_inp, prompt, a_prompt, n_prompt, controlnet_conditioning_scale,
        width, height
    ]
    run_button.click(fn=process, inputs=ips, outputs=video_out)

block.launch(server_name='0.0.0.0', share=True)
