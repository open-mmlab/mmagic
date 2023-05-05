# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
import subprocess
import traceback
import warnings
from typing import Dict, List, Optional, Union

import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from mmengine.registry import init_default_scope

from mmagic.apis.inferencers.inpainting_inferencer import InpaintingInferencer


class InpaintingGradio:
    inpainting_supported_models = [
        # inpainting models
        'aot_gan',
        'deepfillv1',
        'deepfillv2',
        'global_local',
        'partial_conv',
    ]
    inpainting_supported_models_cfg = {}
    inpainting_supported_models_cfg_inited = False
    pkg_path = ''
    error_color = '#FF0000'
    success_color = '#00FF00'
    warning_color = '#FFFF00'
    notice_message = ('', None, '')

    def __init__(self,
                 model_name: str = None,
                 model_setting: int = None,
                 model_config: str = None,
                 model_ckpt: str = None,
                 device: torch.device = None,
                 extra_parameters: Dict = None,
                 seed: int = 2022,
                 **kwargs) -> None:
        init_default_scope('mmagic')
        InpaintingGradio.init_inference_supported_models_cfg()
        self.model_name = model_name
        self.model_setting = model_setting
        self.model_config = model_config
        self.model_ckpt = model_ckpt
        self.device = device
        self.extra_parameters = extra_parameters
        self.seed = seed
        if model_name or (model_config and model_ckpt):
            inpainting_kwargs = {}
            inpainting_kwargs.update(
                self._get_inpainting_kwargs(model_name, model_setting,
                                            model_config, model_ckpt,
                                            extra_parameters))
            self.inference = InpaintingInferencer(
                device=device, seed=seed, **inpainting_kwargs)

    def model_reconfig(self,
                       model_name: str = None,
                       model_setting: int = None,
                       model_config: str = None,
                       model_ckpt: str = None,
                       device: torch.device = None,
                       extra_parameters: Dict = None,
                       seed: int = 2022,
                       **kwargs) -> None:
        inpainting_kwargs = {}
        # if model_config:
        #     model_config = model_config.name
        # if model_ckpt:
        #     model_ckpt = model_ckpt.name
        if not model_name and model_setting:
            self.send_notification(
                'model_name should not be None when model_setting was used',
                self.error_color, 'error')
            return
        elif (not model_config and not model_name) and model_ckpt:
            self.send_notification(
                'model_name and model_config should not be None when '
                'model_ckpt was used', self.error_color, 'error')
            return
        elif (not model_ckpt and not model_name) and model_config:
            self.send_notification(
                'model_name and model_ckpt should not be None when '
                'model_config was used', self.error_color, 'error')
            return
        inpainting_kwargs.update(
            self._get_inpainting_kwargs(model_name, model_setting,
                                        model_config, model_ckpt,
                                        extra_parameters))
        try:
            self.inference = InpaintingInferencer(
                device=device, seed=seed, **inpainting_kwargs)
        except Exception as e:
            self.send_notification('inference Exception:' + str(e),
                                   self.error_color, 'error')
            traceback.print_exc()
            return
        self.send_notification('Model config Finished!', self.success_color,
                               'success')

    @staticmethod
    def change_text2dict(input_text: str) -> Union[Dict, None]:
        return_dict = None
        try:
            return_dict = json.loads(input_text)
        except Exception as e:
            InpaintingGradio.send_notification(
                'Convert string to dict Exception:' + str(e),
                InpaintingGradio.error_color, 'error')
        return return_dict

    @staticmethod
    def get_package_path() -> str:
        p = subprocess.Popen(
            'pip show mmagic', shell=True, stdout=subprocess.PIPE)
        out, err = p.communicate()
        out = out.decode()
        if 'Location' not in out:
            InpaintingGradio.send_notification('module mmagic not found',
                                               InpaintingGradio.error_color,
                                               'error')
            raise Exception('module mmagic not found')
        package_path = out[out.find('Location') +
                           len('Location: '):].split('\r\n')[0] + os.sep
        return package_path

    def get_model_config(self, model_name: str) -> Dict:
        """Get the model configuration including model config and checkpoint
        url.

        Args:
            model_name (str): Name of the model.
        Returns:
            dict: Model configuration.
        """
        if model_name not in self.inpainting_supported_models:
            self.send_notification(f'Model {model_name} is not supported.',
                                   self.error_color, 'error')
            raise ValueError(f'Model {model_name} is not supported.')
        else:
            return self.inpainting_supported_models_cfg[model_name]

    @staticmethod
    def init_inference_supported_models_cfg() -> None:
        if not InpaintingGradio.inpainting_supported_models_cfg_inited:
            InpaintingGradio.pkg_path = InpaintingGradio.get_package_path()
            # all_cfgs_dir = osp.join(osp.dirname(__file__), '..', 'configs')
            all_cfgs_dir = osp.join(InpaintingGradio.pkg_path, 'configs')
            for model_name in InpaintingGradio.inpainting_supported_models:
                meta_file_dir = osp.join(all_cfgs_dir, model_name,
                                         'metafile.yml')
                with open(meta_file_dir, 'r') as stream:
                    parsed_yaml = yaml.safe_load(stream)
                InpaintingGradio.inpainting_supported_models_cfg[
                    model_name] = {}
                InpaintingGradio.inpainting_supported_models_cfg[model_name][
                    'settings'] = parsed_yaml['Models']  # noqa
            InpaintingGradio.inpainting_supported_models_cfg_inited = True

    def _get_inpainting_kwargs(self, model_name: Optional[str],
                               model_setting: Optional[int],
                               model_config: Optional[str],
                               model_ckpt: Optional[str],
                               extra_parameters: Optional[Dict]) -> Dict:
        """Get the kwargs for the inpainting inferencer."""
        kwargs = {}

        if model_name:
            cfgs = self.get_model_config(model_name)
            # kwargs['task'] = cfgs['task']
            setting_to_use = 0
            if model_setting:
                if isinstance(model_setting, str):
                    model_setting = int(
                        model_setting[0:model_setting.find(' - ')])
                setting_to_use = model_setting
            else:
                model_setting = 0
            if model_setting > len(
                    cfgs['settings']) - 1 or model_setting < -len(
                        cfgs['settings']):
                self.send_notification(
                    f"model_setting out of range of {model_name}'s "
                    'cfgs settings', self.error_color, 'error')
            config_dir = cfgs['settings'][setting_to_use]['Config']
            config_dir = config_dir[config_dir.find('configs'):]
            # kwargs['config'] = os.path.join(
            #     osp.dirname(__file__), '..', config_dir)
            kwargs['config'] = os.path.join(self.pkg_path, config_dir)
            kwargs['ckpt'] = cfgs['settings'][setting_to_use]['Weights']

        if model_config:
            if kwargs.get('config', None) is not None:
                warnings.warn(
                    f'{model_name}\'s default config '
                    f'is overridden by {model_config}', UserWarning)
            kwargs['config'] = model_config

        if model_ckpt:
            if kwargs.get('ckpt', None) is not None:
                warnings.warn(
                    f'{model_name}\'s default checkpoint '
                    f'is overridden by {model_ckpt}', UserWarning)
            kwargs['ckpt'] = model_ckpt

        if extra_parameters:
            kwargs['extra_parameters'] = extra_parameters

        return kwargs

    @staticmethod
    def send_notification(msg: str, color: str, label: str) -> None:
        InpaintingGradio.notice_message = (msg, color, label)

    @staticmethod
    def get_inpainting_supported_models() -> List:
        """static function for getting inpainting inference supported modes."""
        return InpaintingGradio.inpainting_supported_models

    def infer(self, input_img_arg: Dict) -> np.ndarray:
        result = self.inference(
            img=input_img_arg['image'], mask=input_img_arg['mask'])
        result = cv2.cvtColor(result[1], cv2.COLOR_RGB2BGR)
        return result

    def run(self) -> None:
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    input_model_dropdown = gr.Dropdown(
                        choices=self.inpainting_supported_models,
                        value=self.model_name,
                        label='choose model')
                    input_setting = gr.Dropdown(
                        value=self.model_setting, label='choose setting')
                    input_config = gr.Textbox(
                        value=self.model_config, label='model_config_path')
                    input_ckpt = gr.Textbox(
                        value=self.model_ckpt, label='model_ckpt_path')
                    input_device_dropdown = gr.Dropdown(
                        choices=['cuda', 'cpu'],
                        label='choose device',
                        value='cuda')
                    input_extra_parameters_input = gr.Textbox(
                        value=json.dumps(self.extra_parameters),
                        label='extra_parameters')
                    input_extra_parameters = gr.JSON(
                        value=self.extra_parameters, label='extra_parameters')
                    input_seed = gr.Number(
                        value=self.seed, precision=0, label='seed')
                    config_button = gr.Button('CONFIG')
                    input_extra_parameters_input.blur(
                        self.change_text2dict, input_extra_parameters_input,
                        input_extra_parameters)

                with gr.Column(visible=False) as output_col:
                    input_image = gr.Image(
                        image_mode='RGB',
                        tool='sketch',
                        type='filepath',
                        label='Input image')
                    infer_button = gr.Button('INFER')
                    infer_button.style(full_width=False)
                    output_image = gr.Image(
                        label='Output image', interactive=False)
                    output_image.style(height=500)
                    infer_button.click(
                        self.infer, inputs=input_image, outputs=output_image)

            with gr.Row():
                label = gr.Label(
                    value=self.notice_message[0],
                    color=self.notice_message[1],
                    label=self.notice_message[2])

            def show_infer(*args) -> Dict:
                self.model_reconfig(*args)
                return {
                    output_col:
                    gr.update(visible=True),
                    label:
                    gr.update(
                        value=self.notice_message[0],
                        color=self.notice_message[1],
                        label=self.notice_message[2])
                }

            def change_setting(model_name: str) -> Dict:
                settings = InpaintingGradio.inpainting_supported_models_cfg[
                    model_name]['settings']
                return {
                    input_setting:
                    gr.update(choices=[
                        str(i) + ' - ' + settings[i]['Config']
                        for i in range(len(settings))
                    ])
                }

            input_model_dropdown.change(
                fn=change_setting,
                inputs=input_model_dropdown,
                outputs=input_setting)

            config_button.click(show_infer, [
                input_model_dropdown,
                input_setting,
                input_config,
                input_ckpt,
                input_device_dropdown,
                input_extra_parameters,
                input_seed,
            ], [output_col, label])
        demo.launch()


if __name__ == '__main__':
    inpaintingGradio = InpaintingGradio()
    inpaintingGradio.run()
