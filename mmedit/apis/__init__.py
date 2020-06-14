from .inpainting_inference import inpainting_inference
from .matting_inference import init_model, matting_inference

__all__ = ['init_model', 'matting_inference', 'inpainting_inference']
