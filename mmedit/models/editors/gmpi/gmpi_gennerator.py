from copy import deepcopy
from typing import Optional, Tuple

import torch
from mmengine.model import BaseModule
from torch import Tensor
import numpy as np

from mmedit.registry import MODULES
from .gmpi_modules import Generator as StyleGAN2Generator
from .renderer import MPIRenderer

@MODULES.register_module('GMPIGenerator')
@MODULES.register_module()
class GMPIGenerator(BaseModule):

    def __init__(self,
                 out_size: int,
                 noise_size: int = 512,
                 style_channels: int = 512,
                 cond_size: int = 25,
                 cond_mapping_channels: Optional[int] = None,
                 cond_scale: float = 1,
                 zero_cond_input: bool = False,
                 num_mlps: int = 8,
                 triplane_size: int = 256,
                 triplane_channels: int = 32,
                 sr_in_size: int = 64,
                 sr_in_channels: int = 32,
                 sr_hidden_channels: int = 128,
                 sr_out_channels: int = 64,
                 sr_antialias: bool = True,
                 sr_add_noise: bool = True,
                 neural_rendering_resolution: int = 64,
                 renderer_cfg: dict = dict(),
                 rgb2bgr: bool = False,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.out_size = out_size
        self.noise_size = noise_size
        self.cond_size = cond_size
        self.style_size = style_channels

        self.cond_scale = cond_scale
        self.zero_cond_input = zero_cond_input
        self.sr_add_noise = sr_add_noise

        # build StyleGAN2 backbone
        self.triplane_channels = triplane_channels
        self.backbone = StyleGAN2Generator(
            out_size=triplane_size,
            noise_size=noise_size,
            out_channels=triplane_channels * 3,
            style_channels=style_channels,
            num_mlps=num_mlps,
            cond_size=cond_size,
            cond_scale=self.cond_scale,
            cond_mapping_channels=cond_mapping_channels,
            zero_cond_input=self.zero_cond_input)

        # build renderer + nerf-decoder and ray sampler
        self.neural_rendering_resolution = neural_rendering_resolution
        renderer_cfg_ = deepcopy(renderer_cfg)
        decoder_cfg_ = renderer_cfg.get('decoder_cfg', dict())
        decoder_cfg_['in_channels'] = triplane_channels
        decoder_cfg_['out_channels'] = sr_in_channels
        renderer_cfg_['decoder_cfg'] = decoder_cfg_
        self.renderer = MPIRenderer(**deepcopy(renderer_cfg_))

        # flag for pretrained models
        self.rgb2bgr = rgb2bgr

    def generate_img(
        self,
        device,
        face_angles,
        generator,
        z,
        mpi_xyz_input,
        metadata,
        horizontal_cam_move=True,
        save_dir=None,
        mpi_xyz_only_z=False,
        z_interpolation_ws=None,
        n_planes=32,
        light_render=None,
        disable_tqdm=False,
        truncation_psi=1.0,
        render_single_image=True,
        chunk_n_planes=-1,
        **kwargs,
    ):

        with torch.no_grad():

            mb_mpi_rgbas = generator(
                z,
                None,
                mpi_xyz_input,
                mpi_xyz_only_z,
                n_planes,
                z_interpolation_ws=z_interpolation_ws,
                truncation_psi=truncation_psi,
            )

            mb_mpi_rgbas = []
            all_n_planes = mpi_xyz_input[4].shape[0]

            if chunk_n_planes == -1:
                chunk_n_planes = all_n_planes + 1

            for tmp_start_idx in range(0, all_n_planes, chunk_n_planes):
                tmp_end_idx = min(all_n_planes, tmp_start_idx + chunk_n_planes)
                tmp_mpi_xyz_input = {}
                for k in mpi_xyz_input:
                    # [#planes, tex_h, tex_w, 3]
                    tmp_mpi_xyz_input[k] = mpi_xyz_input[k][tmp_start_idx:tmp_end_idx, ...]

                tmp_mpi_rgbas = generator(
                    z,
                    None,
                    tmp_mpi_xyz_input,
                    mpi_xyz_only_z,
                    tmp_end_idx - tmp_start_idx,
                    z_interpolation_ws=z_interpolation_ws,
                    truncation_psi=truncation_psi,
                )

                mb_mpi_rgbas.append(tmp_mpi_rgbas)

            mb_mpi_rgbas = torch.cat(mb_mpi_rgbas, dim=1)

            torch.cuda.empty_cache()

            mpi_1st_rgb = mb_mpi_rgbas[:, 0, :3, ...]

            # [#planes, 3, H, W]
            mpi_rgb = mb_mpi_rgbas[0, :, :3, ...]
            # [#planes, 1, H, W]
            mpi_alpha = mb_mpi_rgbas[0, :, 3:, ...]

            tensor_img_list = []
            img_list = []
            depth_img_list = []

            for i, tmp_angle in enumerate(face_angles):

                if render_single_image:
                    metadata["h_mean"] = tmp_angle[0]
                    metadata["v_mean"] = tmp_angle[1]
                else:
                    if horizontal_cam_move:
                        metadata["h_mean"] = tmp_angle
                    else:
                        metadata["v_mean"] = tmp_angle

                img, depth_map, _, _ = self.renderer.render(
                    mb_mpi_rgbas,
                    metadata["img_size"],
                    metadata["img_size"],
                    horizontal_mean=metadata["h_mean"],
                    horizontal_std=metadata["h_stddev"],
                    vertical_mean=metadata["v_mean"],
                    vertical_std=metadata["v_stddev"],
                    assert_not_out_of_last_plane=True,
                )

                tensor_img = img.detach()
                img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                img = (img + 1) / 2.0
                img = (img * 255).astype(np.uint8)

                depth_map = depth_map.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                depth_map = (depth_map - metadata["ray_start"]) / (metadata["ray_end"] - metadata["ray_start"])
                depth_map = np.clip(depth_map, 0, 1)
                depth_map = (depth_map[..., None] * 255).astype(np.uint8)

                tensor_img_list.append(tensor_img)
                img_list.append(img)
                depth_img_list.append(depth_map)

        return img_list, tensor_img_list, depth_img_list, mpi_rgb, mpi_alpha

    def forward(self,
                noise: Tensor,
                label: Optional[Tensor] = None,
                truncation: Optional[float] = 1,
                num_truncation_layer: Optional[int] = None,
                input_is_latent: bool = False,
                plane: Optional[Tensor] = None,
                add_noise: bool = True,
                randomize_noise: bool = True,
                render_kwargs: Optional[dict] = None) -> dict:

        batch_size = noise.shape[0]

        if not input_is_latent:
            styles = self.backbone.mapping(
                noise,
                label,
                truncation=truncation,
                num_truncation_layer=num_truncation_layer)
        else:
            styles = noise

        ray_origins, ray_directions = self.sample_ray(label)

        if plane is None:
            plane = self.backbone.synthesis(
                styles, add_noise=add_noise, randomize_noise=randomize_noise)

        # Reshape output into three `triplane_channels`-channel planes
        plane = plane.view(
            len(plane), 3, self.triplane_channels, plane.shape[-2],
            plane.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, _ = self.renderer(
            plane, ray_origins, ray_directions, render_kwargs=render_kwargs)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            batch_size, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2,
                                            1).reshape(batch_size, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.sr_model(
            rgb_image, feature_image, styles, add_noise=self.sr_add_noise)

        if self.rgb2bgr:
            sr_image = sr_image.flip(1)
            rgb_image = rgb_image.flip(1)

        output_dict = dict(
            fake_img=sr_image,
            lr_img=rgb_image,  # low-resolution images
            depth=depth_image,
            ray_directions=ray_directions,
            ray_origins=ray_origins)

        return output_dict
