# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine import Config
from mmengine.utils import ProgressBar
from torch import Tensor

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import ForwardInputs, SampleList
from ...base_models import BaseConditionalGAN
from ...utils import get_valid_num_batches

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class EG3D(BaseConditionalGAN):
    """Implementation of `Efficient Geometry-aware 3D Generative Adversarial
    Networks`

    <https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_Efficient_Geometry-Aware_3D_Generative_Adversarial_Networks_CVPR_2022_paper.pdf>_ (EG3D).  # noqa

    Detailed architecture can be found in
    :class:`~mmagic.models.editors.eg3d.eg3d_generator.TriplaneGenerator`
    and
    :class:`~mmagic.models.editors.eg3d.dual_discriminator.DualDiscriminator`

    Args:
        generator (ModelType): The config or model of the generator.
        discriminator (Optional[ModelType]): The config or model of the
            discriminator. Defaults to None.
        camera (Optional[ModelType]): The pre-defined camera to sample random
            camera position. If you want to generate images or videos via
            high-level API, you must set this argument. Defaults to None.
        data_preprocessor (Optional[Union[dict, Config]]): The pre-process
            config or :class:`~mmagic.models.DataPreprocessor`.
        generator_steps (int): Number of times the generator was completely
            updated before the discriminator is updated. Defaults to 1.
        discriminator_steps (int): Number of times the discriminator was
            completely updated before the generator is updated. Defaults to 1.
        noise_size (Optional[int]): Size of the input noise vector.
            Default to 128.
        num_classes (Optional[int]): The number classes you would like to
            generate. Defaults to None.
        ema_config (Optional[Dict]): The config for generator's exponential
            moving average setting. Defaults to None.
        loss_config (Optional[Dict]): The config for training losses.
            Defaults to None.
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 camera: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 noise_size: Optional[int] = None,
                 ema_config: Optional[Dict] = None,
                 loss_config: Optional[Dict] = None):

        super().__init__(generator, discriminator, data_preprocessor,
                         generator_steps, discriminator_steps, noise_size,
                         None, ema_config, loss_config)
        if isinstance(camera, dict):
            self.camera = MODELS.build(camera)
        elif isinstance(camera, nn.Module):
            self.camera = camera
        else:
            self.camera = None

    def label_fn(self,
                 label: Optional[Tensor] = None,
                 num_batches: int = 1) -> Tensor:
        """Label sampling function for EG3D model.

        Args:
            label (Optional[Tensor]): Conditional for EG3D model. If not
                passed, :attr:`self.camera` will be used to sample random
                camera-to-world and intrinsics matrix. Defaults to None.

        Returns:
            torch.Tensor: Conditional input for EG3D model.
        """
        if label is not None:
            return label

        # sample random conditional from camera
        assert self.camera is not None, (
            '\'camera\' is not defined for \'EG3D\'.')
        camera2world = self.camera.sample_camera2world(batch_size=num_batches)
        intrinsics = self.camera.sample_intrinsic(batch_size=num_batches)
        cond = torch.cat(
            [camera2world.reshape(-1, 16),
             intrinsics.reshape(-1, 9)], dim=1).to(self.device)
        return cond

    def data_sample_to_label(self, data_sample: SampleList
                             ) -> Optional[torch.Tensor]:
        """Get labels from input `data_sample` and pack to `torch.Tensor`. If
        no label is found in the passed `data_sample`, `None` would be
        returned.

        Args:
            data_sample (List[DataSample]): Input data samples.

        Returns:
            Optional[torch.Tensor]: Packed label tensor.
        """

        if not data_sample or 'gt_label' not in data_sample.keys():
            return None
        return data_sample.gt_label.label

    def pack_to_data_sample(
            self,
            output: Dict[str, Tensor],
            # index: int,
            data_sample: Optional[DataSample] = None) -> DataSample:
        """Pack output to data sample. If :attr:`data_sample` is not passed, a
        new DataSample will be instantiated. Otherwise, outputs will be added
        to the passed datasample.

        Args:
            output (Dict[Tensor]): Output of the model.
            index (int): The index to save.
            data_sample (DataSample, optional): Data sample to save
                outputs. Defaults to None.

        Returns:
            DataSample: Data sample with packed outputs.
        """

        assert isinstance(output,
                          dict), ('Output of EG3D generator should be a dict.')

        data_sample = DataSample() if data_sample is None else data_sample
        for k, v in output.items():
            assert isinstance(v, torch.Tensor), (
                f'Output must be tensor. But \'{k}\' is type of '
                f'\'{type(v)}\'.')
            # NOTE: hard code here, we assume all tensor are [bz, ...]
            data_sample.set_tensor_data({k: v})

        return data_sample

    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None) -> List[DataSample]:
        """Sample images with the given inputs. If forward mode is 'ema' or
        'orig', the image generated by corresponding generator will be
        returned. If forward mode is 'ema/orig', images generated by original
        generator and EMA generator will both be returned in a dict.

        Args:
            inputs (ForwardInputs): Dict containing the necessary
                information (e.g. noise, num_batches, mode) to generate image.
            data_samples (Optional[list]): Data samples collated by
                :attr:`data_preprocessor`. Defaults to None.
            mode (Optional[str]): `mode` is not used in
                :class:`BaseConditionalGAN`. Defaults to None.

        Returns:
            List[DataSample]: Generated images or image dict.
        """
        if isinstance(inputs, Tensor):
            noise = inputs
            sample_kwargs = {}
        else:
            noise = inputs.get('noise', None)
            num_batches = get_valid_num_batches(inputs, data_samples)
            noise = self.noise_fn(noise, num_batches=num_batches)
            sample_kwargs = inputs.get('sample_kwargs', dict())
        num_batches = noise.shape[0]

        labels = self.data_sample_to_label(data_samples)
        if labels is None:
            num_batches = get_valid_num_batches(inputs, data_samples)
            labels = self.label_fn(num_batches=num_batches)

        sample_model = self._get_valid_model(inputs)
        batch_sample_list = []
        if sample_model in ['ema', 'orig']:
            if sample_model == 'ema':
                generator = self.generator_ema
            else:
                generator = self.generator
            outputs = generator(noise, label=labels)
            outputs['fake_img'] = self.data_preprocessor.destruct(
                outputs['fake_img'], data_samples)

            gen_sample = DataSample()
            if data_samples:
                gen_sample.update(data_samples)
            if isinstance(inputs, dict) and 'img' in inputs:
                gen_sample.gt_img = inputs['img']
            gen_sample = self.pack_to_data_sample(outputs, gen_sample)
            gen_sample.noise = noise
            gen_sample.sample_kwargs = deepcopy(sample_kwargs)
            gen_sample.sample_model = sample_model
            batch_sample_list = gen_sample.split(allow_nonseq_value=True)

        else:
            outputs_orig = self.generator(noise, label=labels)
            outputs_ema = self.generator_ema(noise, label=labels)
            outputs_orig['fake_img'] = self.data_preprocessor.destruct(
                outputs_orig['fake_img'], data_samples)
            outputs_ema['fake_img'] = self.data_preprocessor.destruct(
                outputs_ema['fake_img'], data_samples)

            gen_sample = DataSample()
            if data_samples:
                gen_sample.update(data_samples)
            if isinstance(inputs, dict) and 'img' in inputs:
                gen_sample.gt_img = inputs['img']
            gen_sample.ema = self.pack_to_data_sample(outputs_ema)
            gen_sample.orig = self.pack_to_data_sample(outputs_orig)
            gen_sample.noise = noise
            gen_sample.sample_kwargs = deepcopy(sample_kwargs)
            gen_sample.sample_model = sample_model
            batch_sample_list = gen_sample.split(allow_nonseq_value=True)

        return batch_sample_list

    @torch.no_grad()
    def interpolation(self,
                      num_images: int,
                      num_batches: int = 4,
                      mode: str = 'both',
                      sample_model: str = 'orig',
                      show_pbar: bool = True) -> List[dict]:
        """Interpolation input and return a list of output results. We support
        three kinds of interpolation mode:

        * 'camera': First generate style code with random noise and forward
            camera. Then synthesis images with interpolated camera position
            and fixed style code.

        * 'conditioning': First generate style code with fixed noise and
            interpolated camera. Then synthesis images with style codes and
            forward camera.

        * 'both': Generate images with interpolated camera position.

        Args:
            num_images (int): The number of images want to generate.
            num_batches (int, optional): The number of batches to generate at
                one time. Defaults to 4.
            mode (str, optional): The interpolation mode. Supported choices
                are 'both', 'camera', and 'conditioning'. Defaults to 'both'.
            sample_model (str, optional): The model used to generate images,
                support 'orig' and 'ema'. Defaults to 'orig'.
            show_pbar (bool, optional): Whether display a progress bar during
                interpolation. Defaults to True.

        Returns:
            List[dict]: The list of output dict of each frame.
        """
        assert hasattr(self, 'camera'), ('Camera must be defined.')
        assert mode.upper() in ['BOTH', 'CONDITIONING', 'CAMERA']
        assert sample_model in ['ema', 'orig']
        if sample_model == 'orig':
            gen = self.generator
        else:
            assert self.with_ema_gen, (
                '\'sample_model\' is EMA, but ema model not found.')
            # generator_ema is wrapped by AverageModel
            gen = self.generator_ema.module

        cam2world_forward = self.camera.sample_camera2world(
            h_std=0, v_std=0, batch_size=num_batches, device=self.device)
        intrinsics = self.camera.sample_intrinsic(
            batch_size=num_batches, device=self.device)
        cond_forward = torch.cat([
            cam2world_forward.view(num_batches, -1),
            intrinsics.view(num_batches, -1)
        ],
                                 dim=1)
        # 1. generate cond list
        if mode.upper() in ['CAMERA', 'BOTH']:
            cam2world_list = self.camera.interpolation_cam2world(
                num_images, batch_size=num_batches, device=self.device)
            cond_list = []
            for cam2world in cam2world_list:
                cond = torch.cat([
                    cam2world.view(num_batches, -1),
                    intrinsics.view(num_batches, -1)
                ],
                                 dim=1)
                cond_list.append(cond)
        else:
            cond_list = [cond_forward for _ in range(num_images)]

        # 2. generate pre-defined style list
        if mode.upper() == 'CAMERA':
            # same noise + forward cond
            style = gen.backbone.mapping(
                self.noise_fn(num_batches=num_batches), cond_forward)
            style_list = [style for _ in range(num_images)]
        else:
            # same noise + different cond
            noise = self.noise_fn(num_batches=num_batches)
            style_list = [
                gen.backbone.mapping(noise, cond) for cond in cond_list
            ]

        # 3. interpolation
        if show_pbar:
            pbar = ProgressBar(num_images)
        output_list = []
        for style, cond in zip(style_list, cond_list):
            # generate image with const noise
            output = gen(
                style,
                cond,
                input_is_latent=True,
                add_noise=True,
                randomize_noise=False)  # use fixed noise
            output_list.append({k: v.cpu() for k, v in output.items()})

            if show_pbar:
                pbar.update(1)
        if show_pbar:
            print('\n')

        return output_list
