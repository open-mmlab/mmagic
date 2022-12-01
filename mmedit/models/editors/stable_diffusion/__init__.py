from .utils import (
    is_flax_available,
    is_inflect_available,
    is_onnx_available,
    is_scipy_available,
    is_torch_available,
    is_transformers_available,
    is_unidecode_available,
)


__version__ = "0.9.0"

from .configuration_utils import ConfigMixin
from .utils import logging


if is_torch_available():
    from .modeling_utils import ModelMixin
    from .models import AutoencoderKL, Transformer2DModel, UNet1DModel, UNet2DConditionModel, UNet2DModel, VQModel
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from .pipeline_utils import DiffusionPipeline
    from .schedulers import (
        DDIMScheduler,
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KarrasVeScheduler,
        PNDMScheduler,
        RePaintScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
        VQDiffusionScheduler,
    )
    from .training_utils import EMAModel
else:
    from .utils.dummy_pt_objects import *  # noqa F403

if is_torch_available() and is_scipy_available():
    from .schedulers import LMSDiscreteScheduler
else:
    from .utils.dummy_torch_and_scipy_objects import *  # noqa F403

if is_torch_available() and is_transformers_available():
    from .pipelines import (
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionPipeline,
        StableDiffusionUpscalePipeline,
    )
else:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403


from .stable_diffuser import StableDiffuser

__all__ = [
    'StableDiffuser'
]