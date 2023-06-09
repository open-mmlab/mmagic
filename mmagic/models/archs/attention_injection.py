# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
from enum import Enum
from diffusers.models.attention import BasicTransformerBlock

AttentionStatus = Enum('ATTENTION_STATUS', 'READ WRITE')


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class AttentionInjection(nn.Module):
    """Wrapper for stable diffusion unet.

    Args:
        module (nn.Module): The module to be wrapped.
    """

    def __init__(self,
                 module: nn.Module):
        super().__init__()
        self.attention_status = AttentionStatus.READ
        self.style_cfgs = []
        self.unet = module

        attn_inject = self

        def transformer_forward_replacement(self,
                                            hidden_states,
                                            encoder_hidden_states=None,
                                            timestep=None,
                                            attention_mask=None,
                                            cross_attention_kwargs=None,
                                            class_labels=None,
                                            ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(      # noqa
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype)
            else:
                norm_hidden_states = self.norm1(hidden_states)

            attn_output = None
            self_attention_context = norm_hidden_states
            if attn_inject.attention_status == AttentionStatus.WRITE:
                self.bank.append(self_attention_context.detach().clone())
                # self.style_cfgs.append(attn_inject.current_style_fidelity)
            if attn_inject.attention_status == AttentionStatus.READ:
                if len(self.bank) > 0:
                    # style_cfg = sum(self.style_cfgs) / float(
                    #     len(self.style_cfgs))
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=torch.cat(
                            [self_attention_context] + self.bank, dim=1))
                    # attn_output = self.attn1(
                    #     norm_hidden_states,
                    #     encoder_hidden_states=self.bank[0])
                    # self_attn1_c = self_attn1_uc.clone()
                    # self_attn1 = style_cfg * self_attn1_c + \
                    #     (1.0 - style_cfg) * self_attn1_uc
                self.bank = []
                self.style_cfgs = []
            if attn_output is None:
                attn_output = self.attn1(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            cross_attention_kwargs = cross_attention_kwargs if \
                cross_attention_kwargs is not None else {}
            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if
                    self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * \
                    (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        all_modules = torch_dfs(self.unet)

        attn_modules = [module for module in all_modules
                        if isinstance(module, BasicTransformerBlock)]
        for i, module in enumerate(attn_modules):
            if getattr(module, '_original_inner_forward', None) is None:
                module._original_inner_forward = module.forward
            module.forward = transformer_forward_replacement.__get__(
                module, BasicTransformerBlock)
            module.bank = []
            module.style_cfgs = []

    def forward(self,
                x: Tensor,
                t,
                encoder_hidden_states=None,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
                ref_x=None) -> Tensor:
        """Forward and add LoRA mapping.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if ref_x is not None:
            self.attention_status = AttentionStatus.WRITE
            self.unet(
                ref_x,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_additional_residuals,  # noqa
                mid_block_additional_residual=mid_block_additional_residual)
        self.attention_status = AttentionStatus.READ
        output = self.unet(
            x,
            t,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,  # noqa
            mid_block_additional_residual=mid_block_additional_residual)

        return output
