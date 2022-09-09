# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class SearchTransformer(nn.Module):
    """Search texture reference by transformer.

    Include relevance embedding, hard-attention and soft-attention.
    """

    def gather(self, inputs, dim, index):
        """Hard Attention. Gathers values along an axis specified by dim.

        Args:
            inputs (Tensor): The source tensor. (N, C*k*k, H*W)
            dim (int): The axis along which to index.
            index (Tensor): The indices of elements to gather. (N, H*W)

        results:
            outputs (Tensor): The result tensor. (N, C*k*k, H*W)
        """

        views = [inputs.size(0)
                 ] + [1 if i != dim else -1 for i in range(1, inputs.ndim)]
        expansion = [
            -1 if i in (0, dim) else d for i, d in enumerate(inputs.size())
        ]
        index = index.view(views).expand(expansion)
        outputs = torch.gather(inputs, dim, index)

        return outputs

    def forward(self, lq_up, ref_downup, refs):
        """Texture transformer.

        Q = LTE(lq_up)
        K = LTE(ref_downup)
        V = LTE(ref), from V_level_n to V_level_1

        Relevance embedding aims to embed the relevance between the LQ and
            Ref image by estimating the similarity between Q and K.
        Hard-Attention: Only transfer features from the most relevant position
            in V for each query.
        Soft-Attention: synthesize features from the transferred GT texture
            features T and the LQ features F from the backbone.

        Args:
            All args are features come from extractor (such as LTE).
                These features contain 3 levels.
                When upscale_factor=4, the size ratio of these features is
                level3:level2:level1 = 1:2:4.
            lq_up (Tensor): Tensor of 4x bicubic-upsampled lq image.
                (N, C, H, W)
            ref_downup (Tensor): Tensor of ref_downup. ref_downup is obtained
                by applying bicubic down-sampling and up-sampling with factor
                4x on ref. (N, C, H, W)
            refs (Tuple[Tensor]): Tuple of ref tensors.
                [(N, C, H, W), (N, C/2, 2H, 2W), ...]

        Returns:
            soft_attention (Tensor): Soft-Attention tensor. (N, 1, H, W)
            textures (Tuple[Tensor]): Transferred GT textures.
                [(N, C, H, W), (N, C/2, 2H, 2W), ...]
        """

        levels = len(refs)
        # query
        query = F.unfold(lq_up, kernel_size=(3, 3), padding=1)

        # key
        key = F.unfold(ref_downup, kernel_size=(3, 3), padding=1)
        key_t = key.permute(0, 2, 1)

        # values
        values = [
            F.unfold(
                refs[i],
                kernel_size=3 * pow(2, i),
                padding=pow(2, i),
                stride=pow(2, i)) for i in range(levels)
        ]

        key_t = F.normalize(key_t, dim=2)  # [N, H*W, C*k*k]
        query = F.normalize(query, dim=1)  # [N, C*k*k, H*W]

        # Relevance embedding
        rel_embedding = torch.bmm(key_t, query)  # [N, H*W, H*W]
        max_val, max_index = torch.max(rel_embedding, dim=1)  # [N, H*W]

        # hard-attention
        textures = [self.gather(value, 2, max_index) for value in values]

        # to tensor
        h, w = lq_up.size()[-2:]
        textures = [
            F.fold(
                textures[i],
                output_size=(h * pow(2, i), w * pow(2, i)),
                kernel_size=3 * pow(2, i),
                padding=pow(2, i),
                stride=pow(2, i)) / 9. for i in range(levels)
        ]

        soft_attention = max_val.view(max_val.size(0), 1, h, w)

        return soft_attention, textures
