import sys
import cv2
import torch
import numpy as np
import gradio as gr
from diffusers.utils import load_image
from PIL import Image, ImageFilter

torch.set_grad_enabled(False)

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from pipeline.pipeline_PowerPaint_ControlNet import StableDiffusionControlNetInpaintPipeline as Pipeline
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from utils.utils import *

weight_dtype = torch.float16
controlnet_conditioning_scale = 0.5
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype
)
controlnet = controlnet.to("cuda")
global pipe
pipe = Pipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16,safety_checker=None)


pipe.tokenizer = TokenizerWrapper(from_pretrained="runwayml/stable-diffusion-v1-5", subfolder="tokenizer", revision=None)
add_tokens(tokenizer = pipe.tokenizer,text_encoder = pipe.text_encoder,placeholder_tokens = ["MMcontext","MMshape","MMobject"],initialize_tokens = ["a","a","a"],num_vectors_per_token = 10)
pipe.unet.load_state_dict(torch.load("./models/unet/diffusion_pytorch_model.bin"), strict=False)
pipe.text_encoder.load_state_dict(torch.load("./models/text_encoder/pytorch_model.bin"), strict=False)
pipe = pipe.to("cuda")


import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)  
    random.seed(seed) 

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

global current_control
current_control = 'canny'
from controlnet_aux import HEDdetector
from controlnet_aux import OpenposeDetector
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

def predict(input_image, input_control_image, control_type, prompt, ddim_steps, scale, seed,negative_prompt):
    promptA = prompt+" P_obj"
    promptB = prompt+" P_obj"
    negative_promptA = negative_prompt+" P_obj"
    negative_promptB = negative_prompt+" P_obj"
    img = np.array(input_image["image"].convert("RGB"))
    W = int(np.shape(img)[0]-np.shape(img)[0]%8)
    H = int(np.shape(img)[1]-np.shape(img)[1]%8)
    input_image["image"] = input_image["image"].resize((H,W))
    input_image["mask"] = input_image["mask"].resize((H,W))
    print(np.shape(np.array(input_image["mask"].convert("RGB"))))
    print(np.shape(np.array(input_image["image"].convert("RGB"))))

    global current_control
    global pipe
    if current_control != control_type:
        if control_type == 'canny' or control_type is None:            
            pipe.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype)
        elif control_type == 'pose':
            pipe.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=weight_dtype)
        elif control_type == 'depth':
            pipe.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=weight_dtype)
        else:
            pipe.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=weight_dtype)
        pipe = pipe.to("cuda")
        current_control = control_type

    controlnet_image = input_control_image
    if current_control == 'canny':
        controlnet_image = controlnet_image.resize((H, W))
        controlnet_image = np.array(controlnet_image)
        controlnet_image = cv2.Canny(controlnet_image, 100, 200)
        controlnet_image = controlnet_image[:, :, None]
        controlnet_image = np.concatenate([controlnet_image, controlnet_image, controlnet_image], axis=2)
        controlnet_image = Image.fromarray(controlnet_image)
    elif current_control == 'pose':
        controlnet_image = openpose(controlnet_image)
    elif current_control == 'depth':
        controlnet_image = controlnet_image.resize((H, W))
        controlnet_image = get_depth_map(controlnet_image)
    else:
        controlnet_image = hed(controlnet_image)

    mask_np = np.array(input_image["mask"].convert("RGB"))
    controlnet_image= controlnet_image.resize((H,W))
    controlnet_np = np.array(controlnet_image)
    set_seed(seed)
    result = pipe(promptA=promptB,
                    promptB = promptA,
                    tradoff = 1.0,
                    tradoff_nag = 1.0,
                    negative_promptA = negative_promptA,
                    negative_promptB = negative_promptB,
                    image=input_image["image"].convert("RGB"),
                    mask_image=input_image["mask"].convert("RGB"),
                    control_image=controlnet_image,
                    width=H,
                    height=W,
                    guidance_scale = scale,
                    num_inference_steps = ddim_steps).images[0]
    red = np.array(result).astype('float')*1
    red[:,:,0] = 180.0
    red[:,:,2] = 0
    red[:,:,1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray((result_m.astype('float')*(1-mask_np.astype('float')/512.0)+mask_np.astype('float')/512.0*red).astype('uint8'))
    
    controlnet_mask = Image.fromarray((np.array(input_image["image"].convert("RGB")).astype('float')*(1-mask_np.astype('float')/255.0)+mask_np.astype('float')/255.0*controlnet_np).astype('uint8'))

    mask_np = np.array(input_image["mask"].convert("RGB"))
    m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius = 4))
    m_img = np.asarray(m_img)/255.0
    img_np = np.asarray(input_image["image"].convert("RGB"))/255.0
    ours_np = np.asarray(result)/255.0
    ours_np = ours_np*m_img+(1-m_img)*img_np
    result_paste = Image.fromarray(np.uint8(ours_np*255))
    return [input_image["mask"].convert("RGB"),result_m,result,result_paste], [controlnet_image,controlnet_mask]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## PowerPaint with ControlNet")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Input image")
            input_image = gr.Image(source='upload', tool='sketch', type="pil")
            gr.Markdown("## Input control image")
            input_control_image = gr.Image(source='upload', type="pil")
            gr.Markdown("### Input control image")
            control_type = gr.Radio(['canny', 'pose', 'depth', 'hed'])
            promptA = gr.Textbox(label="Prompt")
            negative_promptA = gr.Textbox(label="negative_prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                ddim_steps = gr.Slider(label="Steps", minimum=1,
                                       maximum=50, value=45, step=1)
                scale = gr.Slider(
                    label="Guidance Scale", minimum=0.1, maximum=30.0, value=10, step=0.1
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(
                grid=[2], height="auto")
            control_image_show = gr.Gallery(label="Control condition", show_label=False).style(
                grid=[2], height="auto")
            

    run_button.click(fn=predict, inputs=[
                     input_image, input_control_image, control_type, promptA, ddim_steps, scale, seed,negative_promptA], 
                     outputs=[gallery, control_image_show])


block.launch(share = True,server_name="0.0.0.0",server_port=9586)
