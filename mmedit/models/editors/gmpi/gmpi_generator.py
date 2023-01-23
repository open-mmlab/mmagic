# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from gmpi_modules import MappingNetwork, SynthesisNetwork


# @persistence.persistent_class
class GMPIGenerator(torch.nn.Module):

    def __init__(
        self,
        latent_dim,  # Input latent (Z) dimensionality.
        generator_label_dim,  # Conditioning label (C) dimensionality.
        stylegan2_w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        # img_channels,                  # Number of output color channels.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        synthesis_kwargs={},  # Arguments for SynthesisNetwork.
        # MPI-related
        pos_enc_multires=0,  # Number of channels for positional encoding.
        # ToRGBA-related
        background_alpha_full=False,
        # how to produce MPI's RGB-a
        torgba_sep_background=False,
        # Whether to generate background and foreground separately.
        build_background_from_rgb=False,
        # Whether to build background image from boundaries of RGB.
        build_background_from_rgb_ratio=0.05,
        cond_on_pos_enc_only_alpha=False,
        # Whether to only use "cond_on_pos_enc" for alpha channels.
        gen_alpha_largest_res=256,
        G_final_img_act='none',
    ):
        super().__init__()

        self.step = 0
        self.epoch = 0

        self.latent_dim = latent_dim
        self.generator_label_dim = generator_label_dim
        self.stylegan2_w_dim = stylegan2_w_dim
        self.img_resolution = img_resolution
        self.img_channels = 4

        self.background_alpha_full = background_alpha_full

        self.G_final_img_act = G_final_img_act
        assert self.G_final_img_act in ['none', 'sigmoid',
                                        'tanh'], f'{self.G_final_img_act}'

        self.synthesis = SynthesisNetwork(
            w_dim=stylegan2_w_dim,
            img_resolution=img_resolution,
            img_channels=self.img_channels,
            pos_enc_multires=pos_enc_multires,
            torgba_sep_background=torgba_sep_background,
            build_background_from_rgb=build_background_from_rgb,
            build_background_from_rgb_ratio=build_background_from_rgb_ratio,
            cond_on_pos_enc_only_alpha=cond_on_pos_enc_only_alpha,
            gen_alpha_largest_res=gen_alpha_largest_res,
            **synthesis_kwargs)

        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(
            z_dim=latent_dim,
            c_dim=generator_label_dim,
            w_dim=stylegan2_w_dim,
            num_ws=self.num_ws,
            **mapping_kwargs)

        print('\n[G] total self.num_ws: ', self.num_ws, '\n')

        self.conv_clamp = synthesis_kwargs['conv_clamp']

    def set_tune_toalpha(self):
        for res in self.synthesis.block_resolutions:
            block = getattr(self.synthesis, f'b{res}')
            block.toalpha.requires_grad_(True)

            block.pos_enc_embed.requires_grad_(True)

    def set_tune_tobackground(self):
        for res in self.synthesis.block_resolutions:
            block = getattr(self.synthesis, f'b{res}')
            block.toalpha.requires_grad_(True)
            block.tobackground.requires_grad_(True)

    def synthesize(self,
                   *,
                   ws=None,
                   n_planes=32,
                   mpi_xyz_coords=None,
                   xyz_coords_only_z=False,
                   enable_syn_feat_net_grad=True,
                   **synthesis_kwargs):

        img = self.synthesis(
            ws,
            xyz_coords=mpi_xyz_coords,
            enable_feat_net_grad=enable_syn_feat_net_grad,
            xyz_coords_only_z=xyz_coords_only_z,
            n_planes=n_planes,
            **synthesis_kwargs)
        img = (torch.tanh(img) + 1.0) / 2.0

        bs = img.shape[0]
        full_alpha = torch.ones(
            (bs, 1, self.img_resolution, self.img_resolution),
            device=img.device)
        img = torch.cat((img[:, :-1, ...], full_alpha), dim=1)

        # [B, #planes x 4, H, W] -> [B, #planes, 4, tex_h, tex_w]
        img = img.reshape((img.shape[0], n_planes, 4, self.img_resolution,
                           self.img_resolution))

        return img

    def forward(self,
                z,
                c,
                mpi_xyz_coords,
                xyz_coords_only_z,
                n_planes,
                z_interpolation_ws=None,
                truncation_psi=1,
                truncation_cutoff=None,
                enable_mapping_grad=True,
                enable_syn_feat_net_grad=True,
                **synthesis_kwargs):

        with torch.set_grad_enabled(enable_mapping_grad):
            ws = self.mapping(
                z,
                c,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff)

        img = self.synthesize(
            ws=ws,
            n_planes=n_planes,
            mpi_xyz_coords=mpi_xyz_coords,
            xyz_coords_only_z=xyz_coords_only_z,
            enable_syn_feat_net_grad=enable_syn_feat_net_grad,
            **synthesis_kwargs)

        return img

    def set_device(self, device):
        self.device = device
