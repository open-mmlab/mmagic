# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine import print_log

from mmagic.registry import MODELS


@MODELS.register_module()
class ClipWrapper(nn.Module):
    r"""Clip Models wrapper.

    We provide wrappers for the clip models of ``openai`` and
    ``mlfoundations``, where the user can specify ``clip_type``
    as ``clip`` or ``open_clip``, and then initialize a clip model
    using the same arguments as in the original codebase. The
    following clip models settings are provided in the official
    repo of disco diffusion:
|            Setting            | Source    | Arguments                                                    | # noqa
    |:-----------------------------:|-----------|--------------------------------------------------------------| # noqa
    | ViTB32                        | clip      | name='ViT-B/32',                  jit=False                  | # noqa
    | ViTB16                        | clip      | name='ViT-B/16',                  jit=False                  | # noqa
    | ViTL14                        | clip      | name='ViT-L/14',                  jit=False                  | # noqa
    | ViTL14_336px                  | clip      | name='ViT-L/14@336px',            jit=False                  | # noqa
    | RN50                          | clip      | name='RN50',                      jit=False                  | # noqa
    | RN50x4                        | clip      | name='RN50x4',                    jit=False                  | # noqa
    | RN50x16                       | clip      | name='RN50x16',                   jit=False                  | # noqa
    | RN50x64                       | clip      | name='RN50x64',                   jit=False                  | # noqa
    | RN101                         | clip      | name='RN101',                     jit=False                  | # noqa
    | ViTB32_laion2b_e16            | open_clip | name='ViT-B-32',                  pretrained='laion2b_e16'   | # noqa
    | ViTB32_laion400m_e31          | open_clip | model_name='ViT-B-32',            pretrained='laion400m_e31' | # noqa
    | ViTB32_laion400m_32           | open_clip | model_name='ViT-B-32',            pretrained='laion400m_e32' | # noqa
    | ViTB32quickgelu_laion400m_e31 | open_clip | model_name='ViT-B-32-quickgelu',  pretrained='laion400m_e31' | # noqa
    | ViTB32quickgelu_laion400m_e32 | open_clip | model_name='ViT-B-32-quickgelu',  pretrained='laion400m_e32' | # noqa
    | ViTB16_laion400m_e31          | open_clip | model_name='ViT-B-16',            pretrained='laion400m_e31' | # noqa
    | ViTB16_laion400m_e32          | open_clip | model_name='ViT-B-16',            pretrained='laion400m_e32' | # noqa
    | RN50_yffcc15m                 | open_clip | model_name='RN50',                pretrained='yfcc15m'       | # noqa
    | RN50_cc12m                    | open_clip | model_name='RN50',                pretrained='cc12m'         | # noqa
    | RN50_quickgelu_yfcc15m        | open_clip | model_name='RN50-quickgelu',      pretrained='yfcc15m'       | # noqa
    | RN50_quickgelu_cc12m          | open_clip | model_name='RN50-quickgelu',      pretrained='cc12m'         | # noqa
    | RN101_yfcc15m                 | open_clip | model_name='RN101',               pretrained='yfcc15m'       | # noqa
    | RN101_quickgelu_yfcc15m       | open_clip | model_name='RN101-quickgelu',     pretrained='yfcc15m'       | # noqa

    An example of a ``clip_modes_cfg`` is as follows:

    Examples:

    >>> # Use OpenAI's CLIP
    >>> config = dict(
    >>>     type='ClipWrapper',
    >>>     clip_type='clip',
    >>>     name='ViT-B/32',
    >>>     jit=False)

    >>> # Use OpenCLIP
    >>> config = dict(
    >>>     type='ClipWrapper',
    >>>     clip_type='open_clip',
    >>>     model_name='RN50',
    >>>     pretrained='yfcc15m')

    >>> # Use CLIP from Hugging Face Transformers
    >>> config = dict(
    >>>     type='ClipWrapper',
    >>>     clip_type='huggingface',
    >>>     pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
    >>>     subfolder='text_encoder')

    Args:
        clip_type (List[Dict]): The original source of the clip model. Whether be
            ``clip``, ``open_clip`` or ``hugging_face``.

        *args, **kwargs: Arguments to initialize corresponding clip model.
    """

    def __init__(self, clip_type, *args, **kwargs):

        super().__init__()
        self.clip_type = clip_type
        assert clip_type in ['clip', 'open_clip', 'huggingface']

        error_msg = ('{} need to be installed! Run `pip install -r '
                     'requirements/optional.txt` and try again')
        if clip_type == 'clip':
            try:
                import clip
            except ImportError:
                raise ImportError(error_msg.format('\'clip\''))
            print_log(f'Creating {kwargs["name"]} by OpenAI', 'current')
            self.model, _ = clip.load(*args, **kwargs)
        elif clip_type == 'open_clip':
            try:
                import open_clip
            except ImportError:
                raise ImportError(error_msg.format('\'open_clip_torch\''))
            print_log(f'Creating {kwargs["model_name"]} by '
                      'mlfoundations', 'current')
            self.model = open_clip.create_model(*args, **kwargs)

        elif clip_type == 'huggingface':
            try:
                import transformers
            except ImportError:
                raise ImportError(error_msg.format('\'transforms\''))
            # NOTE: use CLIPTextModel to adopt stable diffusion pipeline
            model_cls = transformers.CLIPTextModel
            self.model = model_cls.from_pretrained(*args, **kwargs)
            self.config = self.model.config
            print_log(
                f'Creating {self.model.name_or_path} '
                'by \'HuggingFace\'', 'current')

        self.model.eval().requires_grad_(False)

    def forward(self, *args, **kwargs):
        """Forward function."""
        return self.model(*args, **kwargs)
