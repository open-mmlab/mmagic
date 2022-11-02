# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union

from mmedit.apis.inferencers import MMEditInferencer
from mmedit.apis.inferencers.base_mmedit_inferencer import InputsType
from mmedit.utils import register_all_modules


class MMEdit:
    """MMEdit API for mmediting models inference.

    Args:
        model_name (str): Name of the editing model. Default to 'FCE_IC15'.
        model_config (str): Path to the config file for the editing model.
            Default to None.
        model_ckpt (str): Path to the checkpoint file for the editing model.
            Default to None.
        config_dir (str): Path to the directory containing config files.
            Default to 'configs/'.
        device (torch.device): Device to use for inference. Default to 'cuda'.
    """

    def __init__(self,
                 model_name: str = None,
                 model_version: str = 'a',
                 model_config: str = None,
                 model_ckpt: str = None,
                 config_dir: str = 'configs/',
                 device: torch.device = 'cuda',
                 **kwargs) -> None:

        register_all_modules(init_default_scope=True)
        self.config_dir = config_dir
        inferencer_kwargs = {}
        inferencer_kwargs.update(
            self._get_inferencer_kwargs(model_name, model_version, model_config, model_ckpt))
        self.inferencer = MMEditInferencer(device=device, **inferencer_kwargs)

    def _get_inferencer_kwargs(self, model: Optional[str], model_version: Optional[str],
                               config: Optional[str], ckpt: Optional[str]) -> Dict:
        """Get the kwargs for the inferencer."""
        kwargs = {}

        if model is not None:
            cfgs = self.get_model_config(model)
            kwargs['type'] = cfgs['type']
            kwargs['config'] = os.path.join(self.config_dir, cfgs['version'][model_version]['config'])
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
        return kwargs

    def infer(self,
                 img: InputsType = None,
                 label: InputsType = None,
                 img_out_dir: str = '',
                 show: bool = False,
                 print_result: bool = False,
                 pred_out_file: str = '',
                 **kwargs) -> Union[Dict, List[Dict]]:
        """Inferences edit model on an image(video) or a
        folder of images(videos).

        Args:
            imgs (str or np.array or Sequence[str or np.array]): Img,
                folder path, np array or list/tuple (with img
                paths or np arrays).
            img_out_dir (str): Output directory of images. Defaults to ''.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            print_result (bool): Whether to print the results.
            pred_out_file (str): File to save the inference results. If left as
                empty, no file will be saved.

        Returns:
            Dict or List[Dict]: Each dict contains the inference result of
            each image. Possible keys are "det_polygons", "det_scores",
            "rec_texts", "rec_scores", "kie_labels", "kie_scores",
            "kie_edge_labels" and "kie_edge_scores".
        """
        return self.inferencer(
            img,
            label,
            img_out_dir=img_out_dir,
            show=show,
            print_result=print_result,
            pred_out_file=pred_out_file)

    def get_model_config(self, model_name: str) -> Dict:
        """Get the model configuration including model config and checkpoint
        url.

        Args:
            model_name (str): Name of the model.
        Returns:
            dict: Model configuration.
        """
        model_dict = {
            # conditional models
            'biggan': {
                'type':
                'conditional',
                'version': {
                    'a': {
                        'config':
                        'biggan/dbnet_resnet18_fpnc_1200e_icdar2015.py',
                        'ckpt':
                        'ckpt/conditional/biggan_cifar10_32x32_b25x2_500k_20210728_110906-08b61a44.pth'  
                    },
                    'b': {
                        'config':
                        'biggan/biggan_ajbrock-sn_8xb32-1500kiters_imagenet1k-128x128.py',
                        'ckpt':
                        'ckpt/conditional/biggan_imagenet1k_128x128_b32x8_best_fid_iter_1232000_20211111_122548-5315b13d.pth'
                    }
                },

            },

            #unconditional models
            'styleganv1': {
                'type':
                'unconditional',
                'config':
                'configs/styleganv1/styleganv1_ffhq-256x256_8xb4-25Mimgs.py',
                'ckpt':
                'styleganv1/styleganv1_ffhq_256_g8_25Mimg_20210407_161748-0094da86.pth'
            }

        }
        if model_name not in model_dict:
            raise ValueError(f'Model {model_name} is not supported.')
        else:
            return model_dict[model_name]
