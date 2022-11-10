# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from typing import Dict, List, Optional, Union

import torch

from mmedit.apis.inferencers import MMEditInferencer
from mmedit.apis.inferencers.base_mmedit_inferencer import InputsType
from mmedit.utils import register_all_modules


class MMEdit:
    """MMEdit API for mmediting models inference.

    Args:
        model_name (str): Name of the editing model.
        model_version (str): Version of a specific model.
            Default to 'a'.
        model_config (str): Path to the config file for the editing model.
            Default to None.
        model_ckpt (str): Path to the checkpoint file for the editing model.
            Default to None.
        config_dir (str): Path to the directory containing config files.
            Default to 'configs/'.
        device (torch.device): Device to use for inference. Default to 'cuda'.
    """
    inference_supported_models = {
        # conditional models
        'biggan': {
            'type': 'conditional',
            'version': {
                'a': {
                    'config':
                    'biggan/biggan_2xb25-500kiters_cifar10-32x32.py',
                    'ckpt':
                    'ckpt/conditional/biggan_cifar10_32x32_b25x2_500k_20210728_110906-08b61a44.pth'  # noqa: E501
                },
                'b': {
                    'config':
                    'biggan/biggan_ajbrock-sn_8xb32-1500kiters_imagenet1k-128x128.py',  # noqa: E501
                    'ckpt':
                    'ckpt/conditional/biggan_imagenet1k_128x128_b32x8_best_fid_iter_1232000_20211111_122548-5315b13d.pth'  # noqa: E501
                }
            },
        },

        # unconditional models
        'styleganv1': {
            'type': 'unconditional',
            'version': {
                'a': {
                    'config':
                    'styleganv1/styleganv1_ffhq-256x256_8xb4-25Mimgs.py',
                    'ckpt':
                    'ckpt/unconditional/styleganv1_ffhq_256_g8_25Mimg_20210407_161748-0094da86.pth'  # noqa: E501
                }
            }
        },

        # matting models
        'gca': {
            'type': 'matting',
            'version': {
                'a': {
                    'config':
                    'gca/gca_r34_4xb10-200k_comp1k.py',
                    'ckpt':
                    'ckpt/matting/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth'  # noqa: E501
                }
            }
        },

        # inpainting models
        'aot_gan': {
            'type': 'inpainting',
            'version': {
                'a': {
                    'config':
                    'aot_gan/aot-gan_smpgan_4xb4_places-512x512.py',
                    'ckpt':
                    'ckpt/inpainting/AOT-GAN_512x512_4x12_places_20220509-6641441b.pth'  # noqa: E501
                }
            }
        },

        # translation models
        'pix2pix': {
            'type': 'translation',
            'version': {
                'a': {
                    'config':
                    'pix2pix/pix2pix_vanilla-unet-bn_1xb1-80kiters_facades.py',  # noqa: E501
                    'ckpt':
                    'ckpt/translation/pix2pix_vanilla_unet_bn_1x1_80k_facades_20210902_170442-c0958d50.pth'  # noqa: E501
                }
            }
        },

        # restoration models
        # real_esrgan error
        'real_esrgan': {
            'type': 'restoration',
            'version': {
                'a': {
                    'config':
                    'real_esrgan/realesrnet_c64b23g32_4xb12-lr2e-4-1000k_df2k-ost.py',  # noqa: E501
                    'ckpt':
                    'ckpt/restoration/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost_20210816-4ae3b5a4.pth'  # noqa: E501
                },
            }
        },
        'esrgan': {
            'type': 'restoration',
            'version': {
                'a': {
                    'config':
                    'esrgan/esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py',  # noqa: E501
                    'ckpt':
                    'ckpt/restoration/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth'  # noqa: E501
                }
            }
        },

        # video_restoration models
        'basicvsr': {
            'type': 'video_restoration',
            'version': {
                'a': {
                    'config':
                    'basicvsr/basicvsr_2xb4_reds4.py',
                    'ckpt':
                    'ckpt/video_restoration/basicvsr_reds4_20120409-0e599677.pth'  # noqa: E501
                },
                'b': {
                    'config':
                    'basicvsr/basicvsr_2xb4_vimeo90k-bi.py',
                    'ckpt':
                    'ckpt/video_restoration/basicvsr_vimeo90k_bi_20210409-d2d8f760.pth'  # noqa: E501
                }
            }
        },

        # video_interpolation models
        'flavr': {
            'type': 'video_interpolation',
            'version': {
                'a': {
                    'config':
                    'flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py',  # noqa: E501
                    'ckpt':
                    'ckpt/video_interpolation/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth'  # noqa: E501
                }
            }
        }
    }

    def __init__(self,
                 model_name: str = None,
                 model_version: str = 'a',
                 model_config: str = None,
                 model_ckpt: str = None,
                 device: torch.device = None,
                 extra_parameters: Dict = None) -> None:
        register_all_modules(init_default_scope=True)
        inferencer_kwargs = {}
        inferencer_kwargs.update(
            self._get_inferencer_kwargs(model_name, model_version,
                                        model_config, model_ckpt,
                                        extra_parameters))
        self.inferencer = MMEditInferencer(device=device, **inferencer_kwargs)

    def _get_inferencer_kwargs(self, model: Optional[str],
                               model_version: Optional[str],
                               config: Optional[str], ckpt: Optional[str],
                               extra_parameters: Optional[Dict]) -> Dict:
        """Get the kwargs for the inferencer."""
        kwargs = {}

        if model is not None:
            cfgs = self.get_model_config(model)
            kwargs['type'] = cfgs['type']
            kwargs['config'] = os.path.join(
                'configs/', cfgs['version'][model_version]['config'])
            kwargs['ckpt'] = cfgs['version'][model_version]['ckpt']
            # kwargs['ckpt'] = 'https://download.openmmlab.com/' + \
            # f'mmediting/{cfgs["version"][model_version]["ckpt"]}'

        if config is not None:
            if kwargs.get('config', None) is not None:
                warnings.warn(
                    f'{model}\'s default config is overridden by {config}',
                    UserWarning)
            kwargs['config'] = config

        if ckpt is not None:
            if kwargs.get('ckpt', None) is not None:
                warnings.warn(
                    f'{model}\'s default checkpoint is overridden by {ckpt}',
                    UserWarning)
            kwargs['ckpt'] = ckpt

        if extra_parameters is not None:
            kwargs['extra_parameters'] = extra_parameters

        return kwargs

    def print_extra_parameters(self):
        """Print the unique parameters of each kind of inferencer."""
        extra_parameters = self.inferencer.get_extra_parameters()
        print(extra_parameters)

    def infer(self,
              img: InputsType = None,
              video: InputsType = None,
              label: InputsType = None,
              trimap: InputsType = None,
              mask: InputsType = None,
              result_out_dir: str = '',
              **kwargs) -> Union[Dict, List[Dict]]:
        """Infer edit model on an image(video).

        Args:
            img (str): Img path.
            video (str): Video path.
            label (int): Label for conditional or unconditional models.
            trimap (str): Trimap path for matting models.
            mask (str): Mask path for inpainting models.
            result_out_dir (str): Output directory of result image or video.
                Defaults to ''.

        Returns:
            Dict or List[Dict]: Each dict contains the inference result of
            each image or video.
        """
        return self.inferencer(
            img=img,
            video=video,
            label=label,
            trimap=trimap,
            mask=mask,
            result_out_dir=result_out_dir,
            **kwargs)

    def get_model_config(self, model_name: str) -> Dict:
        """Get the model configuration including model config and checkpoint
        url.

        Args:
            model_name (str): Name of the model.
        Returns:
            dict: Model configuration.
        """
        if model_name not in self.inference_supported_models:
            raise ValueError(f'Model {model_name} is not supported.')
        else:
            return self.inference_supported_models[model_name]
