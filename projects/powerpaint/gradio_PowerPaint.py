import sys
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from pathlib import Path
torch.set_grad_enabled(False)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from pipeline.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from utils.utils import *
pipe = Pipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16,safety_checker=None)
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

def add_task(prompt,negative_prompt,control_type):
    if control_type == 'Object_removal':
        promptA = prompt+" P_ctxt"
        promptB = prompt+" P_ctxt"
        negative_promptA = negative_prompt+" P_obj"
        negative_promptB = negative_prompt+" P_obj"
    elif control_type == 'Shape_object':
        promptA = prompt+" P_shape"
        promptB = prompt+" P_ctxt"
        negative_promptA = negative_prompt+" P_shape"
        negative_promptB = negative_prompt+" P_ctxt"
    elif control_type == 'Object_inpaint':
        promptA = prompt+" P_obj"
        promptB = prompt+" P_obj"
        negative_promptA = negative_prompt+" P_obj"
        negative_promptB = negative_prompt+" P_obj"

    return promptA,promptB,negative_promptA,negative_promptB

from PIL import Image, ImageFilter
def predict(input_image, mask_img, prompt,Fitting_degree, ddim_steps, scale, seed,negative_prompt,task):
    promptA,promptB,negative_promptA,negative_promptB = add_task(prompt,negative_prompt,task)
    input_image["mask"] = mask_img['image']
    size1,size2 = input_image["image"].convert("RGB").size
    if size1<size2:
        input_image["image"] = input_image["image"].convert("RGB").resize((640,int(size2/size1*640)))
    else:
        input_image["image"] = input_image["image"].convert("RGB").resize((int(size1/size2*640),640))
    img = np.array(input_image["image"].convert("RGB"))

    W = int(np.shape(img)[0]-np.shape(img)[0]%8)
    H = int(np.shape(img)[1]-np.shape(img)[1]%8)
    input_image["image"] = input_image["image"].resize((H,W))
    input_image["mask"] = input_image["mask"].resize((H,W))
    set_seed(seed)
    result = pipe(promptA=promptA,promptB = promptB, tradoff = Fitting_degree,tradoff_nag = Fitting_degree,negative_promptA = negative_promptA,negative_promptB = negative_promptB,image=input_image["image"].convert("RGB"), mask_image=input_image["mask"].convert("RGB"),width=H,height=W,guidance_scale = scale,num_inference_steps = ddim_steps).images[0]
    mask_np = np.array(input_image["mask"].convert("RGB"))
    red = np.array(result).astype('float')*1
    red[:,:,0] = 0
    red[:,:,2] = 180.0
    red[:,:,1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray((result_m.astype('float')*(1-mask_np.astype('float')/512.0)+mask_np.astype('float')/512.0*red).astype('uint8'))
    m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius = 4))
    m_img = np.asarray(m_img)/255.0
    img_np = np.asarray(input_image["image"].convert("RGB"))/255.0
    ours_np = np.asarray(result)/255.0
    ours_np = ours_np*m_img+(1-m_img)*img_np
    result_paste = Image.fromarray(np.uint8(ours_np*255))

    dict_res = [input_image["mask"].convert("RGB"),result_m,result,result_paste]


    return dict_res

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## PowerPaint")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', tool='sketch', type="pil")
            mask_img = gr.Image(source='upload', tool='sketch', type="pil")
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="negative_prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                control_type = gr.Radio(['Object_inpaint', 'Shape_object', 'Object_removal'])
                ddim_steps = gr.Slider(label="Steps", minimum=1,
                                       maximum=50, value=45, step=1)
                scale = gr.Slider(
                    label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1
                )
                Fitting_degree = gr.Slider(
                    label="Fitting degree",
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    randomize=True,
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

    run_button.click(fn=predict, inputs=[
                     input_image,mask_img, prompt,Fitting_degree, ddim_steps, scale, seed,negative_prompt,control_type], outputs=[gallery])
block.launch(share = True,server_name="0.0.0.0",server_port=9586)
