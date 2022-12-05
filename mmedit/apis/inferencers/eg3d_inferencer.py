# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine import print_log
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from PIL import Image
from torch.nn import functional as F
from torchvision.utils import make_grid

from mmedit.structures import EditDataSample
from mmedit.utils import ForwardInputs, try_import
from .base_mmedit_inferencer import BaseMMEditInferencer, InputsType, PredType
from .inference_functions import calculate_grid_size

imageio = try_import('imageio')


class EG3DInferencer(BaseMMEditInferencer):

    func_kwargs = dict(
        preprocess=['inputs'],
        forward=['num_images', 'interpolation'],
        visualize=[
            'result_out_dir', 'vis_mode', 'save_img', 'save_video',
            'img_suffix', 'video_suffix'
        ],
        postprocess=[])

    extra_parameters = dict(num_batches=4, sample_model='ema', add_noise=False)

    def preprocess(self, inputs: InputsType = None) -> ForwardInputs:
        """Process the inputs into a model-feedable format.

        Args:
            inputs (List[Union[str, np.ndarray]]): The conditional inputs for
                the inferencer. Defaults to None.

        Returns:
            ForwardInputs: The preprocessed inputs and data samples.
        """
        if isinstance(inputs, Sequence):
            assert all([type(inputs[0]) == type(lab) for lab in inputs
                        ]), ('All label inputs must have the same type.')
            if isinstance(inputs[0], list):
                for lab in inputs:
                    assert all([isinstance(l_, float) for l_ in lab])
                inputs = np.array(inputs).astype(np.float32)
            elif isinstance(inputs[0], np.ndarray):
                assert all([lab.ndim == 1 for lab in inputs])
                inputs = [input_.astype(np.float32) for input_ in inputs]
            else:
                raise ValueError(
                    'EG3D only support ndarry or list as label input.')

            data_sample_list = []
            for lab in inputs:
                data_sample = EditDataSample()
                data_sample.set_gt_label(lab)
                data_sample_list.append(data_sample.to(self.device))
            self.extra_parameters['num_batches'] = len(inputs)
        else:
            data_sample_list = None

        num_batches = self.extra_parameters['num_batches']
        sample_model = self.extra_parameters['sample_model']
        add_noise = self.extra_parameters['add_noise']
        inputs = dict(
            num_batches=num_batches,
            sample_model=sample_model,
            add_noise=add_noise)
        data_samples = data_sample_list

        return inputs, data_samples

    def forward(self,
                inputs: ForwardInputs,
                interpolation: Optional[str] = 'both',
                num_images: int = 100) -> Union[dict, List[dict]]:
        """Forward the inputs to the model.

        Args:
            inputs (ForwardInputs): Model inputs. If data sample (the second
                element of `inputs`) is not passed, will generate a sequence
                of images corresponding to passed `interpolation` mode.
            interpolation (str): The interplolation mode. Supported choices
                are 'both', 'conditioning', and 'camera'. Defaults to 'both'.
            num_images (int): The number of frames of interpolation.
                Defaults to 500.

        Returns:
            Union[dict, List[dict]]: Output dict corresponds to the input
                condition or the list of output dict of each frame during the
                interpolation process.
        """
        inputs, data_sample = inputs  # unpack the tuple
        # forward as the passed input
        if data_sample is not None:
            outputs = self.model(inputs, data_sample)
            output_dict = defaultdict(list)
            # return outputs
            for output in outputs:
                fake_img = output.fake_img.data
                depth_img = output.depth
                lr_img = output.lr_img.data
                ray_origins = output.ray_origins
                ray_directions = output.ray_directions
                output_dict['fake_img'].append(fake_img)
                output_dict['depth'].append(depth_img)
                output_dict['lr_img'].append(lr_img)
                output_dict['ray_origins'].append(ray_origins)
                output_dict['ray_directions'].append(ray_directions)

            for k in output_dict.keys():
                output_dict[k] = torch.stack(output_dict[k], dim=0)

            return output_dict

        num_batches = inputs['num_batches']
        output_list = self.model.interpolation(num_images, num_batches,
                                               interpolation)
        return output_list

    def visualize(self,
                  preds: Union[PredType, List[PredType]],
                  vis_mode: str = 'both',
                  save_img: bool = True,
                  save_video: bool = True,
                  img_suffix: str = '.png',
                  video_suffix: str = '.mp4',
                  result_out_dir: str = 'eg3d_output') -> None:
        """Visualize predictions.

        Args:
            preds (Union[PredType, List[PredType]]): Prediction os model.
            vis_mode (str, optional): Which output to visualize. Supported
                choices are 'both', 'depth', and 'img'. Defaults to 'all'.
            save_img (bool, optional): Whether save images. Defaults to True.
            save_video (bool, optional): Whether save videos. Defaults to True.
            img_suffix (str, optional): The suffix of saved images.
                Defaults to '.png'.
            video_suffix (str, optional): The suffix of saved videos.
                Defaults to '.mp4'.
            result_out_dir (str, optional): The save director of image and
                videos. Defaults to 'eg3d_output'.
        """
        if save_video:
            assert imageio is not None, (
                'Please install imageio-ffmpeg by \'pip install '
                'imageio-ffmpeg\' to save video.')

        os.makedirs(result_out_dir, exist_ok=True)
        assert vis_mode.upper() in ['BOTH', 'DEPTH', 'IMG']
        if vis_mode.upper() == 'BOTH':
            vis_mode = ['DEPTH', 'IMG']
        if not isinstance(vis_mode, list):
            vis_mode = [vis_mode]

        if not isinstance(preds, list):
            preds = [preds]
            if save_video:
                save_video = False
                print_log('Only one frame of output is generated and cannot '
                          'save video. Set \'save_video\' to \'False\' '
                          'automatically.')
            if not save_img:
                save_img = True
                print_log('Only one frame of output is generated can only save'
                          'image. Set \'save_img\' to \'True\' automatically.')

        # save video
        batch_size = preds[0]['fake_img'].shape[0]

        img_dict = {}
        for target in vis_mode:
            target = 'fake_img' if target.upper() == 'IMG' else target
            if target.lower() == 'fake_img':
                imgs = self.preprocess_img(preds)
            else:
                imgs = self.preprocess_depth(preds)
            img_dict[target.lower()] = imgs

            nrow = calculate_grid_size(batch_size)

            if save_video:
                video_path = osp.join(
                    result_out_dir,
                    f'{target.lower()}_seed{self.seed}{video_suffix}')
                video_writer = imageio.get_writer(
                    video_path,
                    mode='I',
                    fps=60,
                    codec='libx264',
                    bitrate='10M')

            frame_list = torch.split(imgs, batch_size)
            for idx, frame in enumerate(frame_list):
                # frame: [bz, C, H, W]
                frame_grid = make_grid(
                    frame, nrow=nrow).permute(1, 2, 0)[..., (2, 1, 0)]
                frame_grid = frame_grid.numpy().astype(np.uint8)
                if save_video:
                    video_writer.append_data(frame_grid)

                if save_img:
                    if len(frame_list) != 1:
                        img_name = (f'{target.lower()}_frame{idx}_'
                                    f'seed{self.seed}{img_suffix}')
                    else:
                        img_name = (f'{target.lower()}_seed{self.seed}'
                                    f'{img_suffix}')
                    img_path = osp.join(result_out_dir, img_name)
                    Image.fromarray(frame_grid).save(img_path)

            if save_video:
                video_writer.close()
                print_log(f'Save video to \'{video_path}\'.', 'current')

        if len(vis_mode) > 1:
            fake_img = img_dict['fake_img']
            depth_img = img_dict['depth']
            # [num_frame * bz, 3, H, W * 2]
            imgs = torch.cat([fake_img, depth_img], dim=-1)
            nrow = calculate_grid_size(batch_size, aspect_ratio=2)

            if save_video:
                video_path = osp.join(
                    result_out_dir, f'combine_seed{self.seed}{video_suffix}')
                video_writer = imageio.get_writer(
                    video_path,
                    mode='I',
                    fps=60,
                    codec='libx264',
                    bitrate='10M')

            frame_list = torch.split(imgs, batch_size)
            for idx, frame in enumerate(frame_list):
                frame_grid = make_grid(
                    frame, nrow=nrow).permute(1, 2, 0)[..., (2, 1, 0)]
                frame_grid = frame_grid.numpy().astype(np.uint8)

                if save_video:
                    video_writer.append_data(frame_grid)

                if save_img:
                    if len(frame_list) != 1:
                        img_name = (f'combine_frame{idx}_'
                                    f'seed{self.seed}{img_suffix}')
                    else:
                        img_name = (f'combine_seed{self.seed}' f'{img_suffix}')
                    img_path = osp.join(result_out_dir, img_name)
                    Image.fromarray(frame_grid).save(img_path)

            if save_video:
                video_writer.close()
                print_log(f'Save video to \'{video_path}\'.', 'current')

    def preprocess_img(self, preds: List[dict]) -> torch.Tensor:
        """Preprocess images in the predictions.

        Args:
            preds (List[dict]): List of prediction dict of each frame.

        Returns:
            torch.Tensor: Preprocessed image tensor shape like
                [num_frame * bz, 3, H, W].
        """
        imgs = [p['fake_img'].cpu() for p in preds]
        imgs = torch.cat(imgs, dim=0)  # [num_frame * bz, 3, H, W]
        imgs = ((imgs + 1) / 2 * 255.).clamp(0, 255)
        return imgs

    def preprocess_depth(self, preds: List[dict]) -> torch.Tensor:
        """Preprocess depth in the predictions.

        Args:
            preds (List[dict]): List of prediction dict of each frame.

        Returns:
            torch.Tensor: Preprocessed depth tensor shape like
                [num_frame * bz, 3, H, W].
        """
        depth = [p['depth'].cpu() for p in preds]

        depth = torch.cat(depth, dim=0)
        depth = -depth
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.
        depth = depth.clamp(0, 255).repeat(1, 3, 1, 1)

        img_size = preds[0]['fake_img'].shape[-1]
        if img_size != depth.shape[-1]:
            interpolation_kwargs = dict(
                size=img_size, mode='bilinear', align_corners=False)
            if digit_version(TORCH_VERSION) >= digit_version('1.11.0'):
                interpolation_kwargs['antialias'] = True
            depth = F.interpolate(depth, **interpolation_kwargs)
        return depth

    def postprocess(self,
                    preds: PredType,
                    imgs: Optional[List[np.ndarray]] = None,
                    is_batch: bool = False,
                    get_datasample: bool = False) -> Dict[str, torch.tensor]:
        """Postprocess predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            imgs (Optional[np.ndarray]): Visualized predictions.
            is_batch (bool): Whether the inputs are in a batch.
                Defaults to False.
            get_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.

        Returns:
            Dict[str, torch.Tensor]: Inference results as a dict.
        """
        if isinstance(preds[0], dict):
            keys = preds[0].keys()
            outputs = defaultdict(list)
            for pred in preds:
                for k in keys:
                    outputs[k].append(pred[k])
            for k in keys:
                outputs[k] = torch.stack(outputs[k], dim=0)
            return outputs
        # directly return the dict
        return preds
