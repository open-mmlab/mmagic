# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine import print_log

from mmedit.registry import MODELS


@MODELS.register_module()
class ClipWrapper(nn.Module):
    """Clip Models wrapper for disco-diffusion.

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
    .. code-block:: python

        clip_models = [
            dict(type='ClipWrapper', clip_type='clip', name='ViT-B/32', jit=False),
            dict(type='ClipWrapper', clip_type='clip', name='ViT-B/16', jit=False),
            dict(type='ClipWrapper', clip_type='clip', name='RN50', jit=False)
        ]

    Args:
        clip_type (List[Dict]): The original source of the clip model. Whether be
            ``clip`` or ``open_clip``.
    """

    def __init__(self, clip_type, *args, **kwargs):

        super().__init__()
        self.clip_type = clip_type
        assert clip_type in ['clip', 'open_clip']
        if clip_type == 'clip':
            try:
                import clip
            except ImportError:
                raise ImportError(
                    'clip need to be installed! Run `pip install -r requirements/optional.txt` and try again'  # noqa
                )  # noqa
            print_log(f'Creating {kwargs["name"]} by OpenAI', 'current')
            self.model, _ = clip.load(*args, **kwargs)
        elif clip_type == 'open_clip':
            try:
                import open_clip
            except ImportError:
                raise ImportError(
                    'open_clip_torch need to be installed! Run `pip install -r requirements/optional.txt` and try again'  # noqa
                )  # noqa
            print_log(
                f'Creating {kwargs["model_name"]} by mlfoundations',  # noqa
                'current')
            self.model = open_clip.create_model(*args, **kwargs)
        self.model.eval().requires_grad_(False)

    def forward(self, *args, **kwargs):
        """Forward function."""
        return self.model(*args, **kwargs)
