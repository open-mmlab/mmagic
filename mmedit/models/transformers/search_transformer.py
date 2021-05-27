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

    def forward(self, lq_up_level3, ref_downup_level3, ref_level1, ref_level2,
                ref_level3):
        """Texture transformer

        Q = LTE(lq_up)
        K = LTE(ref_downup)
        V = LTE(ref), from V_level1 to V_level3

        Relevance embedding aims to embed the relevance between the LQ and
            Ref image by estimating the similarity between Q and K.
        Hard-Attention: Only transfer features from the most relevant position
            in V for each query.
        Soft-Attention: synthesize features from the transferred GT texture
            features T and the LQ features F from the backbone.

        Args:
            All args are features come from extractor (sucn as LTE).
                These features contain 3 levels.
                When upscale_factor=4, the size ratio of these features is
                level1:level2:level3 = 4:2:1.
            lq_up_level3 (Tensor): level3 feature of 4x bicubic-upsampled lq
                image. (N, 4C, H, W)
            ref_downup_level3 (Tensor): level3 feature of ref_downup.
                ref_downup is obtained by applying bicubic down-sampling and
                up-sampling with factor 4x on ref. (N, 4C, H, W)
            ref_level1 (Tensor): level1 feature of ref image. (N, C, 4H, 4W)
            ref_level2 (Tensor): level2 feature of ref image. (N, 2C, 2H, 2W)
            ref_level3 (Tensor): level3 feature of ref image. (N, 4C, H, W)

        Returns:
            s (Tensor): Soft-Attention tensor. (N, 1, H, W)
            t_level3 (Tensor): Transferred GT texture T in level3.
                (N, 4C, H, W)
            t_level2 (Tensor): Transferred GT texture T in level2.
                (N, 2C, 2H, 2W)
            t_level1 (Tensor): Transferred GT texture T in level1.
                (N, C, 4H, 4W)
        """
        # query
        query = F.unfold(lq_up_level3, kernel_size=(3, 3), padding=1)

        # key
        key = F.unfold(ref_downup_level3, kernel_size=(3, 3), padding=1)
        key_t = key.permute(0, 2, 1)

        # values
        value_level3 = F.unfold(ref_level3, kernel_size=(3, 3), padding=1)
        value_level2 = F.unfold(
            ref_level2, kernel_size=(6, 6), padding=2, stride=2)
        value_level1 = F.unfold(
            ref_level1, kernel_size=(12, 12), padding=4, stride=4)

        key_t = F.normalize(key_t, dim=2)  # [N, H*W, C*k*k]
        query = F.normalize(query, dim=1)  # [N, C*k*k, H*W]

        # Relevance embedding
        rel_embedding = torch.bmm(key_t, query)  # [N, H*W, H*W]
        max_val, max_index = torch.max(rel_embedding, dim=1)  # [N, H*W]

        # hard-attention
        t_level3_unfold = self.gather(value_level3, 2, max_index)
        t_level2_unfold = self.gather(value_level2, 2, max_index)
        t_level1_unfold = self.gather(value_level1, 2, max_index)

        # to tensor
        t_level3 = F.fold(
            t_level3_unfold,
            output_size=lq_up_level3.size()[-2:],
            kernel_size=(3, 3),
            padding=1) / (3. * 3.)
        t_level2 = F.fold(
            t_level2_unfold,
            output_size=(lq_up_level3.size(2) * 2, lq_up_level3.size(3) * 2),
            kernel_size=(6, 6),
            padding=2,
            stride=2) / (3. * 3.)
        t_level1 = F.fold(
            t_level1_unfold,
            output_size=(lq_up_level3.size(2) * 4, lq_up_level3.size(3) * 4),
            kernel_size=(12, 12),
            padding=4,
            stride=4) / (3. * 3.)

        s = max_val.view(
            max_val.size(0), 1, lq_up_level3.size(2), lq_up_level3.size(3))

        return s, t_level3, t_level2, t_level1
