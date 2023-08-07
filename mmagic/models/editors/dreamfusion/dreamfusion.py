# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Optional, Tuple

import torch
from mmengine.model import BaseModel
from mmengine.utils import ProgressBar
from torch import Tensor

from mmagic.models.editors.eg3d.ray_sampler import sample_rays
from mmagic.registry import MODELS
from mmagic.structures import EditDataSample, PixelData


@MODELS.register_module()
class DreamFusion(BaseModel):

    def __init__(self,
                 diffusion,
                 renderer,
                 camera,
                 resolution,
                 data_preprocessor,
                 test_resolution=None,
                 text: Optional[str] = None,
                 negative='',
                 dir_text=True,
                 suppress_face=False,
                 guidance_scale=100,
                 loss_config=dict()):
        super().__init__(data_preprocessor)
        # NOTE: dreamfusion do not need data preprocessor
        self.diffusion = MODELS.build(diffusion)
        self.renderer = MODELS.build(renderer)
        self.camera = MODELS.build(camera)

        self.guidance_scale = guidance_scale
        self.text = text
        self.negative = negative
        self.suppress_face = suppress_face

        self.dir_text = dir_text

        # >>> loss configs
        self.loss_config = deepcopy(loss_config)
        self.weight_entropy = loss_config.get('weight_entropy', 1e-4)
        self.weight_opacity = loss_config.get('weight_opacity', 0)
        self.weight_orient = loss_config.get('weight_orient', 1e-2)
        self.weight_smooth = loss_config.get('weight_smooth', 0)

        self.resolutoin = resolution
        if test_resolution is None:
            self.test_resolution = resolution
        else:
            self.test_resolution = test_resolution

        self.prepare_text_embeddings()

    @property
    def device(self) -> torch.device:
        """Get current device of the model.

        Returns:
            torch.device: The current device of the model.
        """
        return next(self.parameters()).device

    def sample_rays_and_pose(self, num_batches):
        cam2world_matrix, pose = self.camera.sample_camera2world(
            batch_size=num_batches, return_pose=True)
        intrinsics = self.camera.sample_intrinsic(batch_size=num_batches)
        rays_o, rays_d = sample_rays(
            cam2world_matrix, intrinsics, resolution=self.resolutoin)
        return rays_o, rays_d, pose

    def prepare_text_embeddings(self):
        assert self.text is not None, ('\'text\' must be defined in configs '
                                       'or passed by command line args.')

        if not self.dir_text:
            self.text_z = self.diffusion.get_text_embeds([self.text],
                                                         [self.negative])
        else:
            self.text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                # construct dir-encoded text
                text = f'{self.text}, {d} view'

                negative_text = f'{self.negative}'

                # explicit negative dir-encoded text
                if self.suppress_face:
                    if negative_text != '':
                        negative_text += ', '
                    if d == 'back':
                        negative_text += 'face'
                    elif d == 'side':
                        negative_text += 'face'
                    elif d == 'overhead':
                        negative_text += 'face'
                    elif d == 'bottom':
                        negative_text += 'face'

                text_z = self.diffusion.get_text_embeds([text],
                                                        [negative_text])
                self.text_z.append(text_z)

    def label_fn(self, num_batches: int = 1) -> Tuple[Tensor, Tensor]:
        """Label sampling function for DreamFusion model."""
        # sample random conditional from camera
        assert self.camera is not None, (
            '\'camera\' is not defined for \'EG3D\'.')
        camera2world = self.camera.sample_camera2world(batch_size=num_batches)
        intrinsics = self.camera.sample_intrinsic(batch_size=num_batches)

        return camera2world, intrinsics

    def forward(self, inputs, data_samples=None, mode=None):
        # TODO: how to design a better sampler, and what should we return
        num_batches = inputs['num_batches'][0]
        render_kwargs = inputs.get('render_kwargs', dict())
        # TODO: sample a random input, do not support parse input from
        # data_samples currently
        cam2world, intrinsic = self.label_fn(num_batches)

        rays_o, rays_d = sample_rays(
            cam2world, intrinsic, resolution=self.test_resolution)

        # TODO: how can we support other shading mode (e.g., normal)?
        rgb, depth, _ = self.batchify_render(rays_o, rays_d, render_kwargs)
        B, H, W = 1, self.test_resolution, self.test_resolution
        pred_rgb = rgb.reshape(B, H, W, 3)
        pred_rgb = pred_rgb.permute(0, 3, 1, 2)

        pred_depth = depth.reshape(B, H, W, 1).permute(0, 3, 1, 2)
        pred_depth = torch.cat([pred_depth] * 3, dim=1)
        pred_depth = (pred_depth - depth.min()) / (depth.max() - depth.min())

        output = [
            EditDataSample(
                fake_img=PixelData(data=pred_rgb[0]),
                depth=PixelData(data=pred_depth[0]))
        ]

        return output

    @torch.no_grad()
    def interpolation(self,
                      num_images: int,
                      num_batches: int = 1,
                      show_pbar: bool = True):

        assert hasattr(self, 'camera'), ('Camera must be defined.')
        assert num_batches == 1, (
            'DreamFusion only support \'num_batches\' as 1.')
        cam2world_list, pose_list, intrinsic_list = self.camera.interpolation(
            num_images, num_batches)

        output_list = []
        if show_pbar:
            pbar = ProgressBar(num_images)

        for cam2world, pose, intrinsic in zip(cam2world_list, pose_list,
                                              intrinsic_list):

            rays_o, rays_d = sample_rays(
                cam2world, intrinsic, resolution=self.test_resolution)

            rgb, depth, weight = self.batchify_render(rays_o, rays_d)
            B, H, W = 1, self.test_resolution, self.test_resolution
            pred_rgb = rgb.reshape(B, H, W, 3)
            pred_rgb = pred_rgb.permute(0, 3, 1, 2)
            pred_depth = depth.reshape(B, H, W, 1)
            output_list.append(dict(rgb=pred_rgb, depth=pred_depth))

            if show_pbar:
                pbar.update(1)

        if show_pbar:
            print('\n')

        return output_list

    def batchify_render(self, rays_o, rays_d, render_kwarge=dict()):
        # NOTE: can we implement this function with a decorator?
        # If we wrap the renderer, the grad function in train step will not be
        # released

        B, N = rays_o.shape[:2]
        depth = torch.empty((B, N, 1), device=self.device)
        image = torch.empty((B, N, 3), device=self.device)
        weights_sum = torch.empty((B, N, 1), device=self.device)

        max_ray_batch = 4096
        for b in range(B):
            head = 0
            while head < N:
                tail = min(head + max_ray_batch, N)
                rgb_, depth_, weight_ = self.renderer(
                    rays_o[b:b + 1, head:tail],
                    rays_d[b:b + 1, head:tail],
                    render_kwarge,
                )

                depth[b:b + 1, head:tail] = depth_
                weights_sum[b:b + 1, head:tail] = weight_
                image[b:b + 1, head:tail] = rgb_
                head += max_ray_batch

        return image, depth, weights_sum

    def train_step(self, data, optim_wrapper):

        # data preprocessor
        num_batches = data['inputs']['num_batches'][0]
        rays_o, rays_d, pose = self.sample_rays_and_pose(
            num_batches=num_batches)

        B = num_batches
        # N = self.resolutoin**2
        H = W = self.resolutoin

        # forward nerf
        rgb, depth, weight, loss_dict = self.renderer(rays_o, rays_d)
        pred_rgb = rgb.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        pred_rgb = pred_rgb.contiguous()  # [1, 3, H, W]

        # forward diffusion
        if self.dir_text:
            text_z = self.text_z[pose]
        else:
            text_z = self.text_z
        # encode pred_rgb to latents,
        # use train_step to avoid interface conflict
        self.diffusion.module.train_step_(
            text_z, pred_rgb, guidance_scale=self.guidance_scale)

        pred_ws = weight.reshape(B, 1, H, W)

        # NOTE: not used in stable-dreamfusion
        if self.weight_opacity > 0:
            loss_opacity = (pred_ws**2).mean() * self.weight_opacity
            loss_dict['loss_opacity'] = loss_opacity

        # NOTE: author use this to replace opacity one
        if self.weight_entropy > 0:
            alphas = (pred_ws).clamp(1e-5, 1 - 1e-5)
            # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            loss_entropy = (-alphas * torch.log2(alphas) -
                            (1 - alphas) * torch.log2(1 - alphas)).mean()
            loss_entropy = loss_entropy * self.weight_entropy
            loss_dict['loss_entropy'] = loss_entropy

        if 'loss_orient' in loss_dict:
            loss_orient = loss_dict['loss_orient'] * self.weight_orient
            loss_dict['loss_orient'] = loss_orient

        if 'loss_smooth' in loss_dict:
            loss_smooth = loss_dict['loss_smooth'] * self.weight_smooth
            loss_dict['loss_smooth'] = loss_smooth

        loss, log_vars = self.parse_losses(loss_dict)
        optim_wrapper['renderer'].update_params(loss)

        return log_vars

    def test_step(self, data):
        return self.forward(data)

    def val_step(self, data):
        return self.forward(data)
