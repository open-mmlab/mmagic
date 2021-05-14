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
        views = [inputs.size(0)] + [
            1 if i != dim else -1 for i in range(1, len(inputs.size()))
        ]
        expanse = list(inputs.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        outputs = torch.gather(inputs, dim, index)

        return outputs

    def forward(self, lr_pad_lv3, ref_pad_lv3, ref_lv1, ref_lv2, ref_lv3):
        """Texture transformer

        Q = LTE(lr_pad)
        K = LTE(ref_pad)
        V = LTE(ref), from V_lv1 to V_lv3

        Relevance embedding aims to embed the relevance between the LR and
            Ref image by estimating the similarity between Q and K.
        Hard-Attention: Only transfer features from the most relevant position
            in V for each query.
        Soft-Attention: synthesize features from the transferred HR texture
            features T and the LR features F from the backbone.

        Args:
            All args are features come from extractor (sucn as LTE).
                These features contain 3 levels.
                When upscale_factor=4, the size ratio of these features is
                lv1:lv2:lv3 = 4:2:1.
            lr_pad_lv3 (Tensor): Lv3 feature of 4x bicubic-upsampled lq image.
                (N, 4C, H, W)
            ref_pad_lv3 (Tensor): Lv3 feature of ref_pad. Ref_pad is obtained
                by applying bicubic down-sampling and up-sampling with factor
                4x on ref. (N, 4C, H, W)
            ref_lv1 (Tensor): Lv1 feature of ref image. (N, C, 4H, 4W)
            ref_lv2 (Tensor): Lv2 feature of ref image. (N, 2C, 2H, 2W)
            ref_lv3 (Tensor): Lv3 feature of ref image. (N, 4C, H, W)

        Returns:
            s (Tensor): Soft-Attention tensor. (N, 1, H, W)
            t_lv3 (Tensor): Transferred HR texture T in Lv3. (N, 4C, H, W)
            t_lv2 (Tensor): Transferred HR texture T in Lv2. (N, 2C, 2H, 2W)
            t_lv1 (Tensor): Transferred HR texture T in Lv1. (N, C, 4H, 4W)
        """
        # query
        query = F.unfold(lr_pad_lv3, kernel_size=(3, 3), padding=1)

        # key
        key = F.unfold(ref_pad_lv3, kernel_size=(3, 3), padding=1)
        key_t = key.permute(0, 2, 1)

        # values
        value_lv3 = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        value_lv2 = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        value_lv1 = F.unfold(
            ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        key_t = F.normalize(key_t, dim=2)  # [N, Hr*Wr, C*k*k]
        query = F.normalize(query, dim=1)  # [N, C*k*k, H*W]

        # Relevance embedding
        rel_embedding = torch.bmm(key_t, query)  # [N, Hr*Wr, H*W]
        max_val, max_index = torch.max(rel_embedding, dim=1)  # [N, H*W]

        # hard-attention
        t_lv3_unfold = self.gather(value_lv3, 2, max_index)
        t_lv2_unfold = self.gather(value_lv2, 2, max_index)
        t_lv1_unfold = self.gather(value_lv1, 2, max_index)

        # to tensor
        t_lv3 = F.fold(
            t_lv3_unfold,
            output_size=lr_pad_lv3.size()[-2:],
            kernel_size=(3, 3),
            padding=1) / (3. * 3.)
        t_lv2 = F.fold(
            t_lv2_unfold,
            output_size=(lr_pad_lv3.size(2) * 2, lr_pad_lv3.size(3) * 2),
            kernel_size=(6, 6),
            padding=2,
            stride=2) / (3. * 3.)
        t_lv1 = F.fold(
            t_lv1_unfold,
            output_size=(lr_pad_lv3.size(2) * 4, lr_pad_lv3.size(3) * 4),
            kernel_size=(12, 12),
            padding=4,
            stride=4) / (3. * 3.)

        s = max_val.view(
            max_val.size(0), 1, lr_pad_lv3.size(2), lr_pad_lv3.size(3))

        return s, t_lv3, t_lv2, t_lv1
