# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.edit import MMEdit

# yapf: disable

editor = MMEdit(model_name='disco')
text_prompts = {
    0: [
        'clouds surround the mountains and Chinese palaces,sunshine,lake,overlook,overlook,unreal engine,light effect,Dream，Greg Rutkowski,James Gurney,artstation'  # noqa
    ]
}
base_extra_parameters = dict(
    width=1280,
    height=768,
    text_prompts=text_prompts,
    show_progress=True,
    num_inference_steps=250,
    eta=0.8,
    seed=2022)

extra_parameters = base_extra_parameters
editor.infer(
    text=text_prompts,
    result_out_dir='resources/demo_results/disco_results/src.png',
    extra_parameters=extra_parameters)

# image resolution
extra_parameters = base_extra_parameters
extra_parameters['width'] = 768
extra_parameters['height'] = 1280
editor.infer(
    text=text_prompts,
    result_out_dir='resources/demo_results/disco_results/image_resolution.png',
    extra_parameters=extra_parameters)

# clip guidance scale
extra_parameters = base_extra_parameters
extra_parameters['clip_guidance_scale'] = 8000
editor.infer(
    text=text_prompts,
    result_out_dir='resources/demo_results/disco_results/CGS8000.png',
    extra_parameters=extra_parameters)

extra_parameters = base_extra_parameters
extra_parameters['clip_guidance_scale'] = 4000
editor.infer(
    text=text_prompts,
    result_out_dir='resources/demo_results/disco_results/CGS4000.png',
    extra_parameters=extra_parameters)

# yapf: enable
