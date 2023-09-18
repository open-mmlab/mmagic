# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Dict, List, Optional, Union

import torch
import yaml
from mmengine.registry import init_default_scope

from .inferencers import Inferencers
from .inferencers.base_mmagic_inferencer import InputsType


class MMagicInferencer:
    """MMagicInferencer API for mmagic models inference.

    Args:
        model_name (str): Name of the editing model.
        model_setting (str): Setting of a specific model.
            Default to 'a'.
        model_config (str): Path to the config file for the editing model.
            Default to None.
        model_ckpt (str): Path to the checkpoint file for the editing model.
            Default to None.
        config_dir (str): Path to the directory containing config files.
            Default to 'configs/'.
        device (torch.device): Device to use for inference. Default to 'cuda'.

    Examples:
        >>> # inference of a conditional model, biggan for example
        >>> editor = MMagicInferencer(model_name='biggan')
        >>> editor.infer(label=1, result_out_dir='./biggan_res.jpg')

        >>> # inference of a translation model, pix2pix for example
        >>> editor = MMagicInferencer(model_name='pix2pix')
        >>> editor.infer(img='./test.jpg', result_out_dir='./pix2pix_res.jpg')

        >>> # see demo/mmediting_inference_tutorial.ipynb for more examples
    """
    # unsupported now
    # singan, liif
    # output should be checked
    # dic, glean

    inference_supported_models = [
        # colorization models
        'inst_colorization',

        # conditional models
        'biggan',
        'sngan_proj',
        'sagan',

        # unconditional models
        'dcgan',
        'deblurganv2',
        'wgan-gp',
        'lsgan',
        'ggan',
        'pggan',
        'styleganv1',
        'styleganv2',
        'styleganv3',

        # matting models
        'dim',
        'indexnet',
        'gca',

        # inpainting models
        'aot_gan',
        'deepfillv1',
        'deepfillv2',
        'global_local',
        'partial_conv',

        # translation models
        'pix2pix',
        'cyclegan',

        # image super-resolution models
        'srcnn',
        'srgan_resnet',
        'edsr',
        'esrgan',
        'rdn',
        'dic',
        'ttsr',
        'glean',
        'real_esrgan',

        # video_interpolation models
        'flavr',
        'cain',

        # video_restoration models
        'edvr',
        'tdan',
        'basicvsr',
        'iconvsr',
        'basicvsr_pp',
        'real_basicvsr',

        # image_restoration models
        'nafnet',
        'swinir',
        'restormer',

        # text2image models
        'controlnet',
        'disco_diffusion',
        'stable_diffusion',

        # 3D-aware generation
        'eg3d',

        # animation inferencer
        'controlnet_animation',

        # draggan
        'draggan',

        # diffusers pipeline inferencer
        'diffusers_pipeline',
    ]

    inference_supported_models_cfg = {}
    inference_supported_models_cfg_inited = False

    def __init__(self,
                 model_name: str = None,
                 model_setting: int = None,
                 config_name: int = None,
                 model_config: str = None,
                 model_ckpt: str = None,
                 device: torch.device = None,
                 extra_parameters: Dict = None,
                 seed: int = 2022,
                 **kwargs) -> None:
        init_default_scope('mmagic')
        MMagicInferencer.init_inference_supported_models_cfg()
        inferencer_kwargs = {}
        inferencer_kwargs.update(
            self._get_inferencer_kwargs(model_name, model_setting, config_name,
                                        model_config, model_ckpt,
                                        extra_parameters))
        self.inferencer = Inferencers(
            device=device, seed=seed, **inferencer_kwargs)

    def _get_inferencer_kwargs(self, model_name: Optional[str],
                               model_setting: Optional[int],
                               config_name: Optional[int],
                               model_config: Optional[str],
                               model_ckpt: Optional[str],
                               extra_parameters: Optional[Dict]) -> Dict:
        """Get the kwargs for the inferencer."""
        kwargs = {}

        if model_name is not None:
            cfgs = self.get_model_config(model_name)
            kwargs['task'] = cfgs['task']
            setting_to_use = 0
            if model_setting:
                setting_to_use = model_setting
            config_dir = cfgs['settings'][setting_to_use]['Config']
            if config_name:
                for setting in cfgs['settings']:
                    if setting['Name'] == config_name:
                        config_dir = setting['Config']
                        break
            config_dir = config_dir[config_dir.find('configs'):]
            if osp.exists(
                    osp.join(osp.dirname(__file__), '..', '..', config_dir)):
                kwargs['config'] = osp.join(
                    osp.dirname(__file__), '..', '..', config_dir)
            else:
                kwargs['config'] = osp.join(
                    osp.dirname(__file__), '..', '.mim', config_dir)
            if 'Weights' in cfgs['settings'][setting_to_use].keys():
                kwargs['ckpt'] = cfgs['settings'][setting_to_use]['Weights']
                if model_name == 'controlnet':
                    kwargs['ckpt'] = None

        if model_config is not None:
            if kwargs.get('config', None) is not None:
                warnings.warn(
                    f'{model_name}\'s default config '
                    f'is overridden by {model_config}', UserWarning)
            kwargs['config'] = model_config

        if model_ckpt is not None:
            if kwargs.get('ckpt', None) is not None:
                warnings.warn(
                    f'{model_name}\'s default checkpoint '
                    f'is overridden by {model_ckpt}', UserWarning)
            kwargs['ckpt'] = model_ckpt

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
            return self.inference_supported_models_cfg[model_name]

    @staticmethod
    def init_inference_supported_models_cfg() -> None:
        if not MMagicInferencer.inference_supported_models_cfg_inited:
            if osp.exists(
                    osp.join(osp.dirname(__file__), '..', '..', 'configs')):
                all_cfgs_dir = osp.join(
                    osp.dirname(__file__), '..', '..', 'configs')
            else:
                all_cfgs_dir = osp.join(
                    osp.dirname(__file__), '..', '.mim', 'configs')
            for model_name in MMagicInferencer.inference_supported_models:
                meta_file_dir = osp.join(all_cfgs_dir, model_name,
                                         'metafile.yml')
                with open(meta_file_dir, 'r') as stream:
                    parsed_yaml = yaml.safe_load(stream)
                task = parsed_yaml['Models'][0]['Results'][0]['Task']
                MMagicInferencer.inference_supported_models_cfg[
                    model_name] = {}
                MMagicInferencer.inference_supported_models_cfg[model_name][
                    'task'] = task  # noqa
                MMagicInferencer.inference_supported_models_cfg[model_name][
                    'settings'] = parsed_yaml['Models']  # noqa

            MMagicInferencer.inference_supported_models_cfg_inited = True

    @staticmethod
    def get_inference_supported_models() -> List:
        """static function for getting inference supported modes."""
        return MMagicInferencer.inference_supported_models

    @staticmethod
    def get_inference_supported_tasks() -> List:
        """static function for getting inference supported tasks."""
        if not MMagicInferencer.inference_supported_models_cfg_inited:
            MMagicInferencer.init_inference_supported_models_cfg()

        supported_task = set()
        for key in MMagicInferencer.inference_supported_models_cfg.keys():
            if MMagicInferencer.inference_supported_models_cfg[key]['task'] \
               not in supported_task:
                supported_task.add(MMagicInferencer.
                                   inference_supported_models_cfg[key]['task'])
        return list(supported_task)

    @staticmethod
    def get_task_supported_models(task: str) -> List:
        """static function for getting task supported models."""
        if not MMagicInferencer.inference_supported_models_cfg_inited:
            MMagicInferencer.init_inference_supported_models_cfg()

        supported_models = []
        for key in MMagicInferencer.inference_supported_models_cfg.keys():
            if MMagicInferencer.inference_supported_models_cfg[key][
                    'task'] == task:
                supported_models.append(key)
        return supported_models
