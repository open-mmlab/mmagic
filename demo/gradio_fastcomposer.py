# Copyright (c) OpenMMLab. All rights reserved.
import gc

import gradio as gr
import numpy as np
import PIL
import torch
from mmengine import Config

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

gc.collect()
torch.cuda.empty_cache()
register_all_modules()


class ModelWrapper:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def inference(
        self,
        image1: PIL.Image.Image,
        image2: PIL.Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        guidance_scale: float,
        alpha_: float,
        num_steps: int,
        num_images: int,
    ):
        print('Running model inference...')
        image = []
        if image1 is not None:
            image.append(image1)

        if image2 is not None:
            image.append(image2)

        if len(image) == 0:
            return [], 'You need to upload at least one image.'

        num_subject_in_text = (np.array(
            self.model.special_tokenizer.encode(prompt)) ==
                               self.model.image_token_id).sum()
        if num_subject_in_text != len(image):
            return (
                [],
                "Number of subjects in the text description doesn't "
                'match the number of reference images, #text subjects: '
                f'{num_subject_in_text} #reference image: {len(image)}',
            )

        if seed == -1:
            seed = np.random.randint(0, 1000000)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        return (
            self.model.infer(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=512,
                width=512,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
                alpha_=alpha_,
                reference_subject_images=image,
            )['samples'],
            'run successfully',
        )


def create_demo():
    TITLE = 'FastComposer Demo'

    DESCRIPTION = """To run the demo, you should:
    1. Upload your images. The order of image1 and image2 needs to match the
    order of the subects in the prompt. You only need 1 image for
    single subject generation.
    2. Input proper text prompts, such as "A woman img and a man img in
    the snow" or "A painting of a man img in the style of Van Gogh",
    where "img" specifies the token you want to augment and comes
    after the word.
    3. Click the Run button. You can also adjust the hyperparameters to
    improve the results. Look at the job status to see
    if there are any errors with your input.
    """

    cfg_file = Config.fromfile(
        'configs/fastcomposer/fastcomposer_8xb16_FFHQ.py')
    fastcomposer = MODELS.build(cfg_file.model)
    model = ModelWrapper(fastcomposer)
    # model = ModelWrapper(convert_model_to_pipeline(args, "cpu"))
    with gr.Blocks() as demo:
        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    image1 = gr.Image(label='Image 1', type='pil')

                    image2 = gr.Image(label='Image 2', type='pil')

                    gr.Markdown('Upload the image for your subject')

                prompt = gr.Text(
                    value='A man img and a man img sitting in a park',
                    label='Prompt',
                    placeholder=
                    'e.g. "A woman img and a man img in the snow", "A painting'
                    ' of a man img in the style of Van Gogh"',
                    info='Use "img" to specify the word you want to augment.',
                )
                negative_prompt = gr.Text(
                    value='((((ugly)))), (((duplicate))), ((morbid)), '
                    '((mutilated)), [out of frame], extra fingers, '
                    'mutated hands, ((poorly drawn hands)), '
                    '((poorly drawn face)), (((mutation))), '
                    '(((deformed))), ((ugly)), blurry, ((bad anatomy)), '
                    '(((bad proportions))), ((extra limbs)), cloned face, '
                    '(((disfigured))). out of frame, ugly, extra limbs, '
                    '(bad anatomy), gross proportions, (malformed limbs), '
                    '((missing arms)), ((missing legs)), (((extra arms))), '
                    '(((extra legs))), mutated hands, (fused fingers), '
                    '(too many fingers), (((long neck)))',
                    label='Negative Prompt',
                    info='Features that you want to avoid.',
                )
                alpha_ = gr.Slider(
                    label='alpha',
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=0.75,
                    info=
                    'A smaller alpha aligns images with text better, but may '
                    'deviate from the subject image. Increase alpha to improve'
                    ' identity preservation, decrease it for '
                    'prompt consistency.',
                )
                num_images = gr.Slider(
                    label='Number of generated images',
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=4,
                )
                run_button = gr.Button('Run')
                with gr.Accordion(label='Advanced options', open=False):
                    seed = gr.Slider(
                        label='Seed',
                        minimum=-1,
                        maximum=1000000,
                        step=1,
                        value=-1,
                        info='If set to -1, a different seed will be '
                        'used each time.',
                    )
                    guidance_scale = gr.Slider(
                        label='Guidance scale',
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                    )
                    num_steps = gr.Slider(
                        label='Steps',
                        minimum=1,
                        maximum=300,
                        step=1,
                        value=50,
                    )
            with gr.Column():
                result = gr.Gallery(label='Generated Images').style(
                    grid=[2], height='auto')
                error_message = gr.Text(label='Job Status')

        inputs = [
            image1,
            image2,
            prompt,
            negative_prompt,
            seed,
            guidance_scale,
            alpha_,
            num_steps,
            num_images,
        ]
        run_button.click(
            fn=model.inference, inputs=inputs, outputs=[result, error_message])
    return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.queue(api_open=False)
    demo.launch(
        show_error=True,
        server_name='0.0.0.0',
        server_port=8080,
    )
