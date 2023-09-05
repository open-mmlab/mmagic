# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import Transformer2DModel
from diffusers.models.attention import Attention, BasicTransformerBlock
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import BaseOutput, is_torch_version


class ViCoCrossAttnProcessor:
    """Processor for implementing attention for the ViCo method."""

    def __call__(self,
                 attn: Attention,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None):
        """
        Args:
            attn (Attention): Attention module.
            hidden_states (torch.Tensor): Input hidden states.
            encoder_hidden_states (torch.Tensor): Encoder hidden states.
            attention_mask (torch.Tensor): Attention mask.
        Returns:
            torch.Tensor: Output hidden states.
        """
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states \
            if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # new bookkeeping to save the attn probs
        attn.attn_probs = attention_probs

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def replace_cross_attention(unet):
    """Replace Cross Attention processor in UNet."""
    for name, module in unet.named_modules():
        name: str
        if name.endswith('attn2'):
            module.set_processor(ViCoCrossAttnProcessor())


@dataclass
class ViCoTransformer2DModelOutput(BaseOutput):
    """Output for ViCoTransformer2DModel."""
    sample: torch.FloatTensor
    loss_reg: torch.FloatTensor


def otsu(mask_in):
    """Apply otsu for mask.

    Args:
        mask_in (torch.Tensor): Input mask.
    """

    # normalize
    mask_norm = (mask_in - mask_in.min(-1, keepdim=True)[0]) / \
        (mask_in.max(-1, keepdim=True)[0] - mask_in.min(-1, keepdim=True)[0])

    bs = mask_in.shape[0]
    h = mask_in.shape[1]
    mask = []
    for i in range(bs):
        threshold_t = 0.
        max_g = 0.
        for t in range(10):
            mask_i = mask_norm[i]
            low = mask_i[mask_i < t * 0.1]
            high = mask_i[mask_i >= t * 0.1]
            low_num = low.shape[0] / h
            high_num = high.shape[0] / h
            low_mean = low.mean()
            high_mean = high.mean()

            g = low_num * high_num * ((low_mean - high_mean)**2)
            if g > max_g:
                max_g = g
                threshold_t = t * 0.1

        mask_i[mask_i < threshold_t] = 0
        mask_i[mask_i > threshold_t] = 1
        mask.append(mask_i)
    mask_out = torch.stack(mask, dim=0)

    return mask_out


class ViCoTransformer2D(nn.Module):
    """New ViCo-Transformer2D to replace the original Transformer2D model."""

    def __init__(self, org_transformer2d: Transformer2DModel,
                 have_image_cross) -> None:
        """
        Args:
            org_transformer2d (Transformer2DModel): Original
            Transformer2DModel.
            have_image_cross (bool): Flag indicating if the model has
            image_cross_attention modules.
        """
        super().__init__()
        self.transformer_blocks = org_transformer2d.transformer_blocks
        self.is_input_continuous = org_transformer2d.is_input_continuous
        self.norm = org_transformer2d.norm
        self.use_linear_projection = org_transformer2d.use_linear_projection
        self.proj_in = org_transformer2d.proj_in
        self.proj_out = org_transformer2d.proj_out
        self.is_input_vectorized = org_transformer2d.is_input_vectorized
        self.is_input_patches = org_transformer2d.is_input_patches

        num_attention_heads = org_transformer2d.num_attention_heads
        attention_head_dim = org_transformer2d.attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.have_image_cross = have_image_cross
        if self.have_image_cross:
            image_cross_attention = BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                cross_attention_dim=inner_dim)
            self.image_cross_attention = image_cross_attention.to(
                org_transformer2d.device, dtype=org_transformer2d.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        placeholder_position: list = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (
                1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None and (encoder_attention_mask.ndim
                                                   == 2):
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * width, inner_dim)
                hidden_states = self.proj_in(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
            attention_probs = block.attn2.attn_probs[batch // 2:batch, ...]

            loss_reg = None
            if self.have_image_cross:
                # 2.5. image cross attention
                ph_idx, eot_idx = placeholder_position[
                    0], placeholder_position[1]
                attn = attention_probs.transpose(1, 2)
                attn_ph = attn[ph_idx].squeeze(1)  # bs, n_patch
                attn_eot = attn[eot_idx].squeeze(1).detach()

                # ########################
                # attention reg
                if self.image_cross_attention.training:
                    loss_reg = F.mse_loss(
                        attn_ph / attn_ph.max(-1, keepdim=True)[0],
                        attn_eot / attn_eot.max(-1, keepdim=True)[0])
                # ########################

                mask = attn_ph.detach()
                mask = otsu(mask)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1)

                hidden_states, image_reference = hidden_states[:batch // 2], \
                    hidden_states[batch // 2:]
                hidden_states = self.image_cross_attention(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=image_reference,
                    encoder_attention_mask=mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )
                hidden_states = torch.cat([hidden_states, image_reference],
                                          dim=0)

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width,
                                                      inner_dim).permute(
                                                          0, 3, 1,
                                                          2).contiguous()
                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width,
                                                      inner_dim).permute(
                                                          0, 3, 1,
                                                          2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()
        elif self.is_input_patches:
            # TODO: cleanup!
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype)
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(
                2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (
                1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)

            # unpatchify
            height = width = int(hidden_states.shape[1]**0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size,
                       self.out_channels))
            hidden_states = torch.einsum('nhwpqc->nchpwq', hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size,
                       width * self.patch_size))

        if not return_dict:
            return (output, loss_reg)

        return ViCoTransformer2DModelOutput(sample=output, loss_reg=loss_reg)


def replace_transformer2d(module: nn.Module,
                          have_image_cross: Dict[str, List[bool]]):
    """Replace the the Transformer2DModel in UNet.

    Args:
        module (nn.Module): Parent module of Transformer2D.
        have_image_cross (List): List of flag indicating which
        transformer2D modules have image_cross_attention modules.
    """
    down_transformer2d_modules = [(k.rsplit('.', 1), v)
                                  for k, v in module.named_modules()
                                  if isinstance(v, Transformer2DModel)]
    for i, ((parent, k), v) in enumerate(down_transformer2d_modules):
        parent = module.get_submodule(parent)
        setattr(parent, k, ViCoTransformer2D(v, have_image_cross[i]))


class ViCoBlockWrapper(nn.Module):
    """Wrapper for ViCo blocks."""

    def apply_to(self, org_module):
        self.org_module = org_module
        self.org_module.forward = self.forward


class ViCoCrossAttnDownBlock2D(ViCoBlockWrapper):

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        placeholder_position: torch.Tensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        """
        Args:
            hidden_states (torch.FloatTensor): Hidden states.
            temb (Optional[torch.FloatTensor]): Time embedding.
            encoder_hidden_states (Optional[torch.FloatTensor]): Encoder
                hidden states.
            placeholder_position (torch.Tensor): Placeholder position.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            cross_attention_kwargs (Optional[Dict[str, Any]]): Cross attention
                keyword arguments.
            encoder_attention_mask (Optional[torch.FloatTensor]): Encoder
                attention mask.
        Returns:
            torch.FloatTensor: Output hidden states.
            Tuple[torch.FloatTensor]: Output hidden states of each block.
            torch.FloatTensor: Attention regularization loss.
        """
        output_states = ()
        loss_reg_all = 0.0

        for resnet, attn in zip(self.org_module.resnets,
                                self.org_module.attentions):
            attn: ViCoTransformer2D
            if self.org_module.training and (
                    self.org_module.gradient_checkpointing):

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {
                    'use_reentrant': False
                } if is_torch_version('>=', '1.11.0') else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states, loss_reg = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    placeholder_position,
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states, loss_reg = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    placeholder_position=placeholder_position,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )

            output_states = output_states + (hidden_states, )
            if loss_reg is not None:
                loss_reg_all += loss_reg

        if self.org_module.downsamplers is not None:
            for downsampler in self.org_module.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states, )

        return hidden_states, output_states, loss_reg_all


class ViCoUNetMidBlock2DCrossAttn(ViCoBlockWrapper):

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        placeholder_position: torch.Tensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (torch.FloatTensor): Hidden states.
            temb (Optional[torch.FloatTensor]): Time embedding.
            encoder_hidden_states (Optional[torch.FloatTensor]): Encoder
                hidden states.
            placeholder_position (torch.Tensor): Placeholder position.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            cross_attention_kwargs (Optional[Dict[str, Any]]): Cross attention
                keyword arguments.
            encoder_attention_mask (Optional[torch.FloatTensor]): Encoder
                attention mask.
        Returns:
            torch.FloatTensor: Output hidden states.
            torch.FloatTensor: Attention regularization loss.
        """
        loss_reg_all = 0.0
        hidden_states = self.org_module.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.org_module.attentions,
                                self.org_module.resnets[1:]):
            hidden_states, loss_reg = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                placeholder_position=placeholder_position,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )
            hidden_states = resnet(hidden_states, temb)
            if loss_reg is not None:
                loss_reg_all += loss_reg

        return hidden_states, loss_reg_all


class ViCoCrossAttnUpBlock2D(ViCoBlockWrapper):

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        placeholder_position: torch.Tensor = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        """Performs the forward pass through the ViCoCrossAttnUpBlock2D module.

        Args:
            hidden_states (torch.FloatTensor): Input hidden states.
            res_hidden_states_tuple (Tuple[torch.FloatTensor, ...]):
                Tuple of residual hidden states.
            temb (Optional[torch.FloatTensor], optional):
                Temporal embeddings. Defaults to None.
            encoder_hidden_states (Optional[torch.FloatTensor], optional):
                Encoder hidden states. Defaults to None.
            placeholder_position (torch.Tensor, optional):
                Placeholder positions. Defaults to None.
            cross_attention_kwargs (Optional[Dict[str, Any]], optional):
                Keyword arguments for cross-attention. Defaults to None.
            upsample_size (Optional[int], optional): Upsample size.
            attention_mask (Optional[torch.FloatTensor], optional):
              Attention mask.
            encoder_attention_mask (Optional[torch.FloatTensor], optional):
                Encoder attention mask.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]:
                A tuple containing the output hidden states and
                the total regularization loss.
        """
        loss_reg_all = 0.0
        for resnet, attn in zip(self.org_module.resnets,
                                self.org_module.attentions):
            attn: ViCoTransformer2D

            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states],
                                      dim=1)

            if self.org_module.training and (
                    self.org_module.gradient_checkpointing):

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {
                    'use_reentrant': False
                } if is_torch_version('>=', '1.11.0') else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states, loss_reg = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    placeholder_position,
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states, loss_reg = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    placeholder_position=placeholder_position,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )
            if loss_reg is not None:
                loss_reg_all += loss_reg

        if self.org_module.upsamplers is not None:
            for upsampler in self.org_module.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states, loss_reg_all


class ViCoUNet2DConditionOutput(BaseOutput):
    """Output for ViCoUNet2DConditionModel."""
    sample: torch.FloatTensor
    loss_reg: torch.FloatTensor


class ViCoUNet2DConditionModel(ViCoBlockWrapper):
    """UNet2DConditionModel for ViCo Method."""

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        placeholder_position: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """Performs the forward pass through the ViCoBlock2D module.

        Args:
            sample (torch.FloatTensor): Input sample.
            timestep (Union[torch.Tensor, float, int]): Timestep value.
            encoder_hidden_states (torch.Tensor): Encoder hidden states.
            placeholder_position (torch.Tensor): Placeholder positions.
            class_labels (Optional[torch.Tensor], optional): Class labels.
                Defaults to None.
            timestep_cond (Optional[torch.Tensor], optional):
                Timestep condition. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional):
                Attention mask. Defaults to None.
            cross_attention_kwargs (Optional[Dict[str, Any]], optional):
                Keyword arguments for cross-attention. Defaults to None.
            added_cond_kwargs (Optional[Dict[str, torch.Tensor]], optional):
                Additional condition arguments. Defaults to None.
            down_block_additional_residuals
                (Optional[Tuple[torch.Tensor]], optional):
                Additional residuals for down-blocks. Defaults to None.
            mid_block_additional_residual (Optional[torch.Tensor], optional):
                Additional residual for mid-block. Defaults to None.
            encoder_attention_mask (Optional[torch.Tensor], optional):
                Encoder attention mask. Defaults to None.
            return_dict (bool, optional):
                Whether to return a dictionary or a tuple.

        Returns:
            Union[UNet2DConditionOutput, Tuple]:
                The output of the forward pass, which can be either
                a UNet2DConditionOutput object or a tuple of tensors.
        """
        default_overall_up_factor = 2**self.org_module.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.org_module.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == 'mps'
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps],
                                     dtype=dtype,
                                     device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        t_emb = self.org_module.time_proj(timesteps)

        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.org_module.time_embedding(t_emb, timestep_cond)

        if self.org_module.class_embedding is not None:
            if class_labels is None:
                raise ValueError('class_labels should be provided \
                    when num_class_embeds > 0')

            if self.org_module.config.class_embed_type == 'timestep':
                class_labels = self.org_module.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.org_module.class_embedding(class_labels).to(
                dtype=sample.dtype)

            if self.org_module.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.org_module.config.addition_embed_type == 'text':
            aug_emb = self.org_module.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb
        elif self.org_module.config.addition_embed_type == 'text_image':
            # Kadinsky 2.1 - style
            if 'image_embeds' not in added_cond_kwargs:
                raise ValueError(
                    f"{self.org_module.__class__} has the config param \
                    `addition_embed_type` set to 'text_image' which \
                    requires the keyword argument `image_embeds` \
                    to be passed in `added_cond_kwargs`")

            image_embs = added_cond_kwargs.get('image_embeds')
            text_embs = added_cond_kwargs.get('text_embeds',
                                              encoder_hidden_states)

            aug_emb = self.org_module.add_embedding(text_embs, image_embs)
            emb = emb + aug_emb

        if self.org_module.time_embed_act is not None:
            emb = self.org_module.time_embed_act(emb)

        if self.org_module.encoder_hid_proj is not None and (
                self.org_module.config.encoder_hid_dim_type == 'text_proj'):
            encoder_hidden_states = self.org_module.encoder_hid_proj(
                encoder_hidden_states)
        elif self.org_module.encoder_hid_proj is not None and (
                self.org_module.config.encoder_hid_dim_type
                == 'text_image_proj'):
            # Kadinsky 2.1 - style
            if 'image_embeds' not in added_cond_kwargs:
                raise ValueError(
                    f"{self.org_module.__class__} has the config param \
                        `encoder_hid_dim_type` set to 'text_image_proj' which \
                        requires the keyword argument `image_embeds` to be \
                        passed in  `added_conditions`")

            image_embeds = added_cond_kwargs.get('image_embeds')
            encoder_hidden_states = self.org_module.encoder_hid_proj(
                encoder_hidden_states, image_embeds)

        # 2. pre-process
        sample = self.org_module.conv_in(sample)

        loss_reg_all = 0.0 if self.training else None

        # 3. down
        down_block_res_samples = (sample, )
        for downsample_block in self.org_module.down_blocks:
            if hasattr(downsample_block, 'has_cross_attention'
                       ) and downsample_block.has_cross_attention:
                sample, res_samples, loss_reg = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    placeholder_position=placeholder_position,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
                if self.training:
                    loss_reg_all += loss_reg
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals):
                down_block_res_sample = down_block_res_sample + \
                    down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (
                    down_block_res_sample, )

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.org_module.mid_block is not None:
            sample, loss_reg = self.org_module.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
            if self.training:
                loss_reg_all += loss_reg
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.org_module.up_blocks):
            is_final_block = i == len(self.org_module.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, 'has_cross_attention'
                       ) and upsample_block.has_cross_attention:
                sample, loss_reg = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    placeholder_position=placeholder_position,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                if self.training:
                    loss_reg_all += loss_reg
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size)

        # 6. post-process
        if self.org_module.conv_norm_out:
            sample = self.org_module.conv_norm_out(sample)
            sample = self.org_module.conv_act(sample)
        sample = self.org_module.conv_out(sample)

        if not return_dict:
            return (sample, loss_reg_all)

        return ViCoUNet2DConditionOutput(sample=sample, loss_reg=loss_reg_all)


def set_vico_modules(unet, image_cross_layers):
    """Set all modules for ViCo method after the UNet initialized normally.

    Args:
        unet (nn.Module): UNet model.
        image_cross_layers (List): List of flag indicating which
        transformer2D modules have image_cross_attention modules.
    """
    # replace transformer2d blocks
    replace_transformer2d(unet, image_cross_layers)

    # replace cross attention layer
    replace_cross_attention(unet)

    # replace forward
    for _, layer in unet.named_modules():
        if layer.__class__.__name__ == 'UNet2DConditionModel':
            vico_unet = ViCoUNet2DConditionModel()
            vico_unet.apply_to(unet)
        elif layer.__class__.__name__ == 'CrossAttnDownBlock2D':
            vico_down_block = ViCoCrossAttnDownBlock2D()
            vico_down_block.apply_to(layer)
        elif layer.__class__.__name__ == 'UNetMidBlock2DCrossAttn':
            vico_mid_block = ViCoUNetMidBlock2DCrossAttn()
            vico_mid_block.apply_to(layer)
        elif layer.__class__.__name__ == 'CrossAttnUpBlock2D':
            vico_up_block = ViCoCrossAttnUpBlock2D()
            vico_up_block.apply_to(layer)
