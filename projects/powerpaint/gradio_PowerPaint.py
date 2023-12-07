import random

import gradio as gr
import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageFilter
from pipeline.pipeline_PowerPaint import \
    StableDiffusionInpaintPipeline as Pipeline
from transformers import CLIPTextModel, CLIPTokenizer
from utils.utils import TokenizerWrapper, add_tokens

torch.set_grad_enabled(False)

weight_type = torch.float16
pipe = Pipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting',
    torch_dtype=torch.float16,
    safety_checker=None)
pipe.tokenizer = CLIPTokenizer.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    subfolder='tokenizer',
    revision=None,
    torch_dtype=torch.float16)
pipe.text_encoder = CLIPTextModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    subfolder='text_encoder',
    revision=None,
    torch_dtype=torch.float16)
pipe.vae = AutoencoderKL.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    subfolder='vae',
    revision=None,
    torch_dtype=torch.float16)
pipe.tokenizer = TokenizerWrapper(
    from_pretrained='runwayml/stable-diffusion-v1-5',
    subfolder='tokenizer',
    revision=None)

add_tokens(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    placeholder_tokens=['MMcontext', 'MMshape', 'MMobject'],
    initialize_tokens=['a', 'a', 'a'],
    num_vectors_per_token=10)
pipe.unet.load_state_dict(
    torch.load(
        '/mnt/petrelfs/share_data/zhuangjunhao/checkpoint-22000/diffusion_pytorch_model.bin'  # noqa
    ),
    strict=False)
pipe.text_encoder.load_state_dict(
    torch.load(
        '/mnt/petrelfs/share_data/zhuangjunhao/checkpoint-22000/change_pytorch_model.bin'  # noqa
    ),
    strict=False)
pipe = pipe.to('cuda')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_task(prompt, negative_prompt, control_type):
    if control_type == 'context-aware':
        promptA = prompt + ' MMcontext'
        promptB = prompt + ' MMcontext'
        negative_promptA = negative_prompt + ' MMcontext'
        negative_promptB = negative_prompt + ' MMcontext'
    elif control_type == 'object-removal':
        promptA = prompt + ' MMcontext'
        promptB = prompt + ' MMcontext'
        negative_promptA = negative_prompt + ' MMobject'
        negative_promptB = negative_prompt + ' MMobject'
    elif control_type == 'shape-guided':
        promptA = prompt + ' MMshape'
        promptB = prompt + ' MMcontext'
        negative_promptA = negative_prompt + ' MMshape'
        negative_promptB = negative_prompt + ' MMcontext'
    elif control_type == 'text-guided':
        promptA = prompt + ' MMobject'
        promptB = prompt + ' MMobject'
        negative_promptA = negative_prompt + ' MMobject'
        negative_promptB = negative_prompt + ' MMobject'
    else:
        promptA = prompt + ' MMcontext'
        promptB = prompt + ' MMcontext'
        negative_promptA = negative_prompt + ' MMcontext'
        negative_promptB = negative_prompt + ' MMcontext'

    return promptA, promptB, negative_promptA, negative_promptB


def predict(input_image, prompt, fitting_degree, ddim_steps, scale, seed,
            negative_prompt, task):
    promptA, promptB, negative_promptA, negative_promptB = add_task(
        prompt, negative_prompt, task)
    size1, size2 = input_image['image'].convert('RGB').size
    if size1 < size2:
        input_image['image'] = input_image['image'].convert('RGB').resize(
            (640, int(size2 / size1 * 640)))
    else:
        input_image['image'] = input_image['image'].convert('RGB').resize(
            (int(size1 / size2 * 640), 640))
    img = np.array(input_image['image'].convert('RGB'))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image['image'] = input_image['image'].resize((H, W))
    input_image['mask'] = input_image['mask'].resize((H, W))
    set_seed(seed)
    result = pipe(
        promptA=promptA,
        promptB=promptB,
        tradoff=fitting_degree,
        tradoff_nag=fitting_degree,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image['image'].convert('RGB'),
        mask_image=input_image['mask'].convert('RGB'),
        width=H,
        height=W,
        guidance_scale=scale,
        num_inference_steps=ddim_steps).images[0]
    mask_np = np.array(input_image['mask'].convert('RGB'))
    red = np.array(result).astype('float') * 1
    red[:, :, 0] = 0
    red[:, :, 2] = 180.0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (result_m.astype('float') * (1 - mask_np.astype('float') / 512.0) +
         mask_np.astype('float') / 512.0 * red).astype('uint8'))
    m_img = input_image['mask'].convert('RGB').filter(
        ImageFilter.GaussianBlur(radius=4))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image['image'].convert('RGB')) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))

    dict_res = [input_image['mask'].convert('RGB'), result_m]

    return result_paste, dict_res


with gr.Blocks(css='style.css') as demo:
    with gr.Row():
        gr.Markdown(
            "<div align='center'><font size='18'>PowerPaint: High-Quality Versatile Image Inpainting</font></div>") # noqa
    with gr.Row():
        gr.Markdown(
            "<div align='center'><font size='5'><a href='https://powerpaint.github.io/'>Project Page</a> &ensp;" # noqa
            "<a href='https://arxiv.org/abs/2312.03594/'>Paper</a> &ensp;"
            "<a href='https://github.com/open-mmlab/mmagic/tree/main/projects/powerpaint'>Code</a> </font></div>"  # noqa
        )

    with gr.Row():
        with gr.Column():
            gr.Markdown('### Input image and draw mask')
            input_image = gr.Image(source='upload', tool='sketch', type='pil')

            gr.Markdown('### Input prompt')
            prompt = gr.Textbox(label='Prompt')
            negative_prompt = gr.Textbox(label='negative_prompt')

            gr.Markdown('### Select task')
            control_type = gr.Radio([
                'context-aware', 'text-guided', 'object-removal',
                'shape-guided'
            ])

            gr.Markdown('### Select fitting degree if task is shape-guided')
            fitting_degree = gr.Slider(
                label='fitting degree',
                minimum=0,
                maximum=1,
                step=0.05,
                randomize=True,
            )
            run_button = gr.Button(label='Run')
            with gr.Accordion('Advanced options', open=False):
                ddim_steps = gr.Slider(
                    label='Steps', minimum=1, maximum=50, value=45, step=1)
                scale = gr.Slider(
                    label='Guidance Scale',
                    minimum=0.1,
                    maximum=30.0,
                    value=7.5,
                    step=0.1)
                seed = gr.Slider(
                    label='Seed',
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )
        with gr.Column():
            gr.Markdown('### Inpainting result')
            inpaint_result = gr.Image()
            gr.Markdown('### Mask')
            gallery = gr.Gallery(
                label='Generated images', show_label=False).style(
                    grid=[2], height='auto')

    run_button.click(
        fn=predict,
        inputs=[
            input_image, prompt, fitting_degree, ddim_steps, scale, seed,
            negative_prompt, control_type
        ],
        outputs=[inpaint_result, gallery])

demo.queue()
demo.launch(share=False, server_name='0.0.0.0', server_port=7890)
