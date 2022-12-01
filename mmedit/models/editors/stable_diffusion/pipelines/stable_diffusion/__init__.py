from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

import PIL
from PIL import Image

from ...utils import BaseOutput

@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


from .pipeline_cycle_diffusion import CycleDiffusionPipeline
from .pipeline_stable_diffusion import StableDiffusionPipeline
from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from .pipeline_stable_diffusion_inpaint_legacy import StableDiffusionInpaintPipelineLegacy
from .pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
from .safety_checker import StableDiffusionSafetyChecker

