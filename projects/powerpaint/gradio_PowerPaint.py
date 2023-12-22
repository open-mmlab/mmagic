import random

import cv2
import gradio as gr
import numpy as np
import torch
from controlnet_aux import HEDdetector, OpenposeDetector
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from PIL import Image, ImageFilter
from pipeline.pipeline_PowerPaint import \
    StableDiffusionInpaintPipeline as Pipeline
from pipeline.pipeline_PowerPaint_ControlNet import \
    StableDiffusionControlNetInpaintPipeline as controlnetPipeline
from safetensors.torch import load_model
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from utils.utils import TokenizerWrapper, add_tokens

torch.set_grad_enabled(False)

weight_dtype = torch.float16
global pipe
pipe = Pipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting', torch_dtype=weight_dtype)
pipe.tokenizer = TokenizerWrapper(
    from_pretrained='runwayml/stable-diffusion-v1-5',
    subfolder='tokenizer',
    revision=None)

add_tokens(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    placeholder_tokens=['P_ctxt', 'P_shape', 'P_obj'],
    initialize_tokens=['a', 'a', 'a'],
    num_vectors_per_token=10)

load_model(pipe.unet, './models/unet/diffusion_pytorch_model.safetensors')
pipe.text_encoder.load_state_dict(
    torch.load('./models/text_encoder/pytorch_model.bin'), strict=False)
pipe = pipe.to('cuda')

depth_estimator = DPTForDepthEstimation.from_pretrained(
    'Intel/dpt-hybrid-midas').to('cuda')
feature_extractor = DPTFeatureExtractor.from_pretrained(
    'Intel/dpt-hybrid-midas')
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

global current_control
current_control = 'canny'
# controlnet_conditioning_scale = 0.8


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_depth_map(image):
    image = feature_extractor(
        images=image, return_tensors='pt').pixel_values.to('cuda')
    with torch.no_grad(), torch.autocast('cuda'):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode='bicubic',
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def add_task(prompt, negative_prompt, control_type):
    # print(control_type)
    if control_type == 'object-removal':
        promptA = prompt + ' P_ctxt'
        promptB = prompt + ' P_ctxt'
        negative_promptA = negative_prompt + ' P_obj'
        negative_promptB = negative_prompt + ' P_obj'
    elif control_type == 'shape-guided':
        promptA = prompt + ' P_shape'
        promptB = prompt + ' P_ctxt'
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
    elif control_type == 'image-outpainting':
        promptA = prompt + ' P_ctxt'
        promptB = prompt + ' P_ctxt'
        negative_promptA = negative_prompt + ' P_obj'
        negative_promptB = negative_prompt + ' P_obj'
    else:
        promptA = prompt + ' P_obj'
        promptB = prompt + ' P_obj'
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt

    return promptA, promptB, negative_promptA, negative_promptB


def predict(input_image, prompt, fitting_degree, ddim_steps, scale, seed,
            negative_prompt, task, vertical_expansion_ratio,
            horizontal_expansion_ratio):
    size1, size2 = input_image['image'].convert('RGB').size

    if task != 'image-outpainting':
        if size1 < size2:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (640, int(size2 / size1 * 640)))
        else:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (int(size1 / size2 * 640), 640))
    else:
        if size1 < size2:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (512, int(size2 / size1 * 512)))
        else:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (int(size1 / size2 * 512), 512))

    if (vertical_expansion_ratio is not None
            and horizontal_expansion_ratio is not None):
        o_W, o_H = input_image['image'].convert('RGB').size
        c_W = int(horizontal_expansion_ratio * o_W)
        c_H = int(vertical_expansion_ratio * o_H)

        expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * 127
        original_img = np.array(input_image['image'])
        expand_img[int((c_H - o_H) / 2.0):int((c_H - o_H) / 2.0) + o_H,
                   int((c_W - o_W) / 2.0):int((c_W - o_W) / 2.0) +
                   o_W, :] = original_img

        blurry_gap = 10

        expand_mask = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255
        if vertical_expansion_ratio == 1 and horizontal_expansion_ratio != 1:
            expand_mask[int((c_H - o_H) / 2.0):int((c_H - o_H) / 2.0) + o_H,
                        int((c_W - o_W) / 2.0) +
                        blurry_gap:int((c_W - o_W) / 2.0) + o_W -
                        blurry_gap, :] = 0  # noqa
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio != 1:
            expand_mask[int((c_H - o_H) / 2.0) +
                        blurry_gap:int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                        int((c_W - o_W) / 2.0) +
                        blurry_gap:int((c_W - o_W) / 2.0) + o_W -
                        blurry_gap, :] = 0  # noqa
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio == 1:
            expand_mask[int((c_H - o_H) / 2.0) +
                        blurry_gap:int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                        int((c_W - o_W) /
                            2.0):int((c_W - o_W) / 2.0) + o_W, :] = 0  # noqa

        input_image['image'] = Image.fromarray(expand_img)
        input_image['mask'] = Image.fromarray(expand_mask)

    promptA, promptB, negative_promptA, negative_promptB = add_task(
        prompt, negative_prompt, task)
    print(promptA, promptB, negative_promptA, negative_promptB)
    img = np.array(input_image['image'].convert('RGB'))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image['image'] = input_image['image'].resize((H, W))
    input_image['mask'] = input_image['mask'].resize((H, W))
    set_seed(seed)
    global pipe
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
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (result_m.astype('float') * (1 - mask_np.astype('float') / 512.0) +
         mask_np.astype('float') / 512.0 * red).astype('uint8'))
    m_img = input_image['mask'].convert('RGB').filter(
        ImageFilter.GaussianBlur(radius=3))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image['image'].convert('RGB')) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))

    dict_res = [input_image['mask'].convert('RGB'), result_m]

    dict_out = [input_image['image'].convert('RGB'), result_paste]

    return dict_out, dict_res


def predict_controlnet(input_image, input_control_image, control_type, prompt,
                       ddim_steps, scale, seed, negative_prompt,
                       controlnet_conditioning_scale):
    promptA = prompt + ' P_obj'
    promptB = prompt + ' P_obj'
    negative_promptA = negative_prompt
    negative_promptB = negative_prompt
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

    global current_control
    global pipe

    base_control = ControlNetModel.from_pretrained(
        'lllyasviel/sd-controlnet-canny', torch_dtype=weight_dtype)
    control_pipe = controlnetPipeline(pipe.vae, pipe.text_encoder,
                                      pipe.tokenizer, pipe.unet, base_control,
                                      pipe.scheduler, None, None, False)
    control_pipe = control_pipe.to('cuda')
    current_control = 'canny'
    if current_control != control_type:
        if control_type == 'canny' or control_type is None:
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-canny', torch_dtype=weight_dtype)
        elif control_type == 'pose':
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-openpose', torch_dtype=weight_dtype)
        elif control_type == 'depth':
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-depth', torch_dtype=weight_dtype)
        else:
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-hed', torch_dtype=weight_dtype)
        control_pipe = control_pipe.to('cuda')
        current_control = control_type

    controlnet_image = input_control_image
    if current_control == 'canny':
        controlnet_image = controlnet_image.resize((H, W))
        controlnet_image = np.array(controlnet_image)
        controlnet_image = cv2.Canny(controlnet_image, 100, 200)
        controlnet_image = controlnet_image[:, :, None]
        controlnet_image = np.concatenate(
            [controlnet_image, controlnet_image, controlnet_image], axis=2)
        controlnet_image = Image.fromarray(controlnet_image)
    elif current_control == 'pose':
        controlnet_image = openpose(controlnet_image)
    elif current_control == 'depth':
        controlnet_image = controlnet_image.resize((H, W))
        controlnet_image = get_depth_map(controlnet_image)
    else:
        controlnet_image = hed(controlnet_image)

    mask_np = np.array(input_image['mask'].convert('RGB'))
    controlnet_image = controlnet_image.resize((H, W))
    set_seed(seed)
    result = control_pipe(
        promptA=promptB,
        promptB=promptA,
        tradoff=1.0,
        tradoff_nag=1.0,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image['image'].convert('RGB'),
        mask_image=input_image['mask'].convert('RGB'),
        control_image=controlnet_image,
        width=H,
        height=W,
        guidance_scale=scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=ddim_steps).images[0]
    red = np.array(result).astype('float') * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (result_m.astype('float') * (1 - mask_np.astype('float') / 512.0) +
         mask_np.astype('float') / 512.0 * red).astype('uint8'))

    mask_np = np.array(input_image['mask'].convert('RGB'))
    m_img = input_image['mask'].convert('RGB').filter(
        ImageFilter.GaussianBlur(radius=4))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image['image'].convert('RGB')) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))
    return [input_image['image'].convert('RGB'),
            result_paste], [controlnet_image, result_m]


def infer(input_image, text_guided_prompt, text_guided_negative_prompt,
          shape_guided_prompt, shape_guided_negative_prompt, fitting_degree,
          ddim_steps, scale, seed, task, enable_control, input_control_image,
          control_type, vertical_expansion_ratio, horizontal_expansion_ratio,
          outpaint_prompt, outpaint_negative_prompt,
          controlnet_conditioning_scale, removal_prompt,
          removal_negative_prompt):
    if task == 'text-guided':
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt
    elif task == 'shape-guided':
        prompt = shape_guided_prompt
        negative_prompt = shape_guided_negative_prompt
    elif task == 'object-removal':
        prompt = removal_prompt
        negative_prompt = removal_negative_prompt
    elif task == 'image-outpainting':
        prompt = outpaint_prompt
        negative_prompt = outpaint_negative_prompt
        return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
                       seed, negative_prompt, task, vertical_expansion_ratio,
                       horizontal_expansion_ratio)
    else:
        task = 'text-guided'
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt

    if enable_control and task == 'text-guided':
        return predict_controlnet(input_image, input_control_image,
                                  control_type, prompt, ddim_steps, scale,
                                  seed, negative_prompt,
                                  controlnet_conditioning_scale)
    else:
        return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
                       seed, negative_prompt, task, None, None)


def select_tab_text_guided():
    return 'text-guided'


def select_tab_object_removal():
    return 'object-removal'


def select_tab_image_outpainting():
    return 'image-outpainting'


def select_tab_shape_guided():
    return 'shape-guided'


with gr.Blocks(css='style.css') as demo:
    with gr.Row():
        gr.Markdown(
            "<div align='center'><font size='18'>PowerPaint: High-Quality Versatile Image Inpainting</font></div>"  # noqa
        )
    with gr.Row():
        gr.Markdown(
            "<div align='center'><font size='5'><a href='https://powerpaint.github.io/'>Project Page</a> &ensp;"  # noqa
            "<a href='https://arxiv.org/abs/2312.03594/'>Paper</a> &ensp;"
            "<a href='https://github.com/open-mmlab/mmagic/tree/main/projects/powerpaint'>Code</a> </font></div>"  # noqa
        )
    with gr.Row():
        gr.Markdown(
            '**Note:** Due to network-related factors, the page may experience occasional bugs！ If the inpainting results deviate significantly from expectations, consider toggling between task options to refresh the content.'  # noqa
        )
    with gr.Row():
        with gr.Column():
            gr.Markdown('### Input image and draw mask')
            input_image = gr.Image(source='upload', tool='sketch', type='pil')

            task = gr.Radio([
                'text-guided', 'object-removal', 'shape-guided',
                'image-outpainting'
            ],
                            show_label=False,
                            visible=False)

            # Text-guided object inpainting
            with gr.Tab('Text-guided object inpainting') as tab_text_guided:
                enable_text_guided = gr.Checkbox(
                    label='Enable text-guided object inpainting',
                    value=True,
                    interactive=False)
                text_guided_prompt = gr.Textbox(label='Prompt')
                text_guided_negative_prompt = gr.Textbox(
                    label='negative_prompt')
                gr.Markdown('### Controlnet setting')
                enable_control = gr.Checkbox(
                    label='Enable controlnet',
                    info='Enable this if you want to use controlnet')
                controlnet_conditioning_scale = gr.Slider(
                    label='controlnet conditioning scale',
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=0.5,
                )
                control_type = gr.Radio(['canny', 'pose', 'depth', 'hed'],
                                        label='Control type')
                input_control_image = gr.Image(source='upload', type='pil')
            tab_text_guided.select(
                fn=select_tab_text_guided, inputs=None, outputs=task)

            # Object removal inpainting
            with gr.Tab('Object removal inpainting') as tab_object_removal:
                enable_object_removal = gr.Checkbox(
                    label='Enable object removal inpainting',
                    value=True,
                    info='The recommended configuration for '
                    'the Guidance Scale is 10 or higher.'
                    'If undesired objects appear in the masked area, '
                    'you can address this by specifically increasing '
                    'the Guidance Scale.',
                    interactive=False)
                removal_prompt = gr.Textbox(label='Prompt')
                removal_negative_prompt = gr.Textbox(label='negative_prompt')
            tab_object_removal.select(
                fn=select_tab_object_removal, inputs=None, outputs=task)

            # Object image outpainting
            with gr.Tab('Image outpainting') as tab_image_outpainting:
                enable_object_removal = gr.Checkbox(
                    label='Enable image outpainting',
                    value=True,
                    info='The recommended configuration for the Guidance '
                    'Scale is 10 or higher. '
                    'If unwanted random objects appear in '
                    'the extended image region, '
                    'you can enhance the cleanliness of the extension '
                    'area by increasing the Guidance Scale.',
                    interactive=False)
                outpaint_prompt = gr.Textbox(label='Outpainting_prompt')
                outpaint_negative_prompt = gr.Textbox(
                    label='Outpainting_negative_prompt')
                horizontal_expansion_ratio = gr.Slider(
                    label='horizontal expansion ratio',
                    minimum=1,
                    maximum=4,
                    step=0.05,
                    value=1,
                )
                vertical_expansion_ratio = gr.Slider(
                    label='vertical expansion ratio',
                    minimum=1,
                    maximum=4,
                    step=0.05,
                    value=1,
                )
            tab_image_outpainting.select(
                fn=select_tab_image_outpainting, inputs=None, outputs=task)

            # Shape-guided object inpainting
            with gr.Tab('Shape-guided object inpainting') as tab_shape_guided:
                enable_shape_guided = gr.Checkbox(
                    label='Enable shape-guided object inpainting',
                    value=True,
                    interactive=False)
                shape_guided_prompt = gr.Textbox(label='shape_guided_prompt')
                shape_guided_negative_prompt = gr.Textbox(
                    label='shape_guided_negative_prompt')
                fitting_degree = gr.Slider(
                    label='fitting degree',
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=1,
                )
            tab_shape_guided.select(
                fn=select_tab_shape_guided, inputs=None, outputs=task)

            run_button = gr.Button(label='Run')
            with gr.Accordion('Advanced options', open=False):
                ddim_steps = gr.Slider(
                    label='Steps', minimum=1, maximum=50, value=45, step=1)
                scale = gr.Slider(
                    label='Guidance Scale',
                    info='For object removal and image outpainting, '
                    'it is recommended to set the value at 10 or above.',
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
            inpaint_result = gr.Gallery(
                label='Generated images', show_label=False, columns=2)
            gr.Markdown('### Mask')
            gallery = gr.Gallery(
                label='Generated masks', show_label=False, columns=2)

    run_button.click(
        fn=infer,
        inputs=[
            input_image, text_guided_prompt, text_guided_negative_prompt,
            shape_guided_prompt, shape_guided_negative_prompt, fitting_degree,
            ddim_steps, scale, seed, task, enable_control, input_control_image,
            control_type, vertical_expansion_ratio, horizontal_expansion_ratio,
            outpaint_prompt, outpaint_negative_prompt,
            controlnet_conditioning_scale, removal_prompt,
            removal_negative_prompt
        ],
        outputs=[inpaint_result, gallery])

demo.queue()
demo.launch(share=False, server_name='0.0.0.0', server_port=7860)
