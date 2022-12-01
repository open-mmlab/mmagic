from ..utils import is_flax_available, is_onnx_available, is_torch_available, is_transformers_available

if is_torch_available() and is_transformers_available():
    from .stable_diffusion import (
        CycleDiffusionPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionPipeline,
        StableDiffusionUpscalePipeline,
    )


