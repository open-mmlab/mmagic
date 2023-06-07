# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
from enum import Enum
from diffusers.models.transformer_2d import Transformer2DModel

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

        all_modules = torch_dfs(self.unet)

        attn_modules = [module for module in all_modules
                        if isinstance(module, Transformer2DModel)]
        attn_modules = sorted(attn_modules,
                              key=lambda x: - x.norm1.normalized_shape[0])

        for i, module in enumerate(attn_modules):
            if getattr(module, '_original_inner_forward', None) is None:
                module._original_inner_forward = module._forward
            module._forward = self.transformer_forward_replacement.__get__(
                module, Transformer2DModel)
            module.bank = []
            module.style_cfgs = []

    def forward(self, x: Tensor, ref_x: Tensor) -> Tensor:
        """Forward and add LoRA mapping.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if ref_x is not None:
            self.attention_status = AttentionStatus.WRITE
            self.unet(ref_x)
        self.attention_status = AttentionStatus.READ
        output = self.unet(x)

        return output

    def transformer_forward_replacement(self, x, context=None):
        x_norm1 = self.norm1(x)
        self_attn1 = None
        if self.disable_self_attn:
            # Do not use self-attention
            self_attn1 = self.attn1(x_norm1, context=context)
        else:
            # Use self-attention
            self_attention_context = x_norm1
            if self.attention_status == AttentionStatus.Write:
                self.bank.append(self_attention_context.detach().clone())
                self.style_cfgs.append(self.current_style_fidelity)
            if self.attention_status == AttentionStatus.Read:
                if len(self.bank) > 0:
                    style_cfg = \
                        sum(self.style_cfgs) / float(len(self.style_cfgs))
                    self_attn1_uc = self.attn1(
                        x_norm1,
                        context=torch.cat(
                            [self_attention_context] + self.bank, dim=1))
                    self_attn1_c = self_attn1_uc.clone()
                    self_attn1 = style_cfg * self_attn1_c + \
                        (1.0 - style_cfg) * self_attn1_uc
                self.bank = []
                self.style_cfgs = []
            if self_attn1 is None:
                self_attn1 = self.attn1(
                    x_norm1, context=self_attention_context)

        x = self_attn1.to(x.dtype) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
