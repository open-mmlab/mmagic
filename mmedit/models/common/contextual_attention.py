from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualAttentionModule(nn.Module):
    """Contexture attention module.

    The details of this module can be found in:
    Generative Image Inpainting with Contextual Attention

    Args:
        unfold_raw_kernel_size (int): Kernel size used in unfolding raw
            feature. Default: 4.
        unfold_raw_stride (int): Stride used in unfolding raw feature. Default:
            2.
        unfold_raw_padding (int): Padding used in unfolding raw feature.
            Default: 1.
        unfold_corr_kernel_size (int): Kernel size used in unfolding
            context for computing correlation maps. Default: 3.
        unfold_corr_stride (int): Stride used in unfolding context for
            computing correlation maps. Default: 1.
        unfold_corr_dilation (int): Dilation used in unfolding context for
            computing correlation maps. Default: 1.
        unfold_corr_padding (int): Padding used in unfolding context for
            computing correlation maps. Default: 1.
        scale (float): The resale factor used in resize input features.
            Default: 0.5.
        fuse_kernel_size (int): The kernel size used in fusion module.
            Default: 3.
        softmax_scale (float): The scale factor for softmax function.
            Default: 10.
        return_attenion_score (bool): If True, the attention score will be
            returned. Default: True.
    """

    def __init__(self,
                 unfold_raw_kernel_size=4,
                 unfold_raw_stride=2,
                 unfold_raw_padding=1,
                 unfold_corr_kernel_size=3,
                 unfold_corr_stride=1,
                 unfold_corr_dilation=1,
                 unfold_corr_padding=1,
                 scale=0.5,
                 fuse_kernel_size=3,
                 softmax_scale=10,
                 return_attenion_score=True):
        super(ContextualAttentionModule, self).__init__()
        self.unfold_raw_kernel_size = unfold_raw_kernel_size
        self.unfold_raw_stride = unfold_raw_stride
        self.unfold_raw_padding = unfold_raw_padding
        self.unfold_corr_kernel_size = unfold_corr_kernel_size
        self.unfold_corr_stride = unfold_corr_stride
        self.unfold_corr_dilation = unfold_corr_dilation
        self.unfold_corr_padding = unfold_corr_padding
        self.scale = scale
        self.fuse_kernel_size = fuse_kernel_size
        self.with_fuse_correlation = fuse_kernel_size > 1
        self.softmax_scale = softmax_scale
        self.return_attention_score = return_attenion_score

        if self.with_fuse_correlation:
            assert fuse_kernel_size % 2 == 1
            fuse_kernel = torch.eye(fuse_kernel_size).view(
                1, 1, fuse_kernel_size, fuse_kernel_size)
            self.register_buffer('fuse_kernel', fuse_kernel)
            padding = int((fuse_kernel_size - 1) // 2)
            self.fuse_conv = partial(F.conv2d, padding=padding, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, context, mask=None):
        """Forward Function.

        Args:
            x (torch.Tensor): Tensor with shape (n, c, h, w).
            context (torch.Tensor): Tensor with shape (n, c, h, w).
            mask (torch.Tensor): Tensor with shape (n, 1, h, w). Default: None.

        Returns:
            tuple(torch.Tensor): Features after contextural attention.
        """
        # raw features to be used in copy (deconv)
        raw_context = context
        raw_context_cols = self.im2col(
            raw_context,
            kernel_size=self.unfold_raw_kernel_size,
            stride=self.unfold_raw_stride,
            padding=self.unfold_raw_padding,
            normalize=False,
            return_cols=True)
        # resize the feature to reduce computational cost
        x = F.interpolate(x, scale_factor=self.scale)
        context = F.interpolate(context, scale_factor=self.scale)

        context_cols = self.im2col(
            context,
            kernel_size=self.unfold_corr_kernel_size,
            stride=self.unfold_corr_stride,
            padding=self.unfold_corr_padding,
            dilation=self.unfold_corr_dilation,
            normalize=True,
            return_cols=True)
        h_unfold, w_unfold = self.calculate_unfold_hw(
            context.size()[-2:],
            kernel_size=self.unfold_corr_kernel_size,
            stride=self.unfold_corr_stride,
            padding=self.unfold_corr_padding,
            dilation=self.unfold_corr_dilation,
        )
        # reshape context_cols to
        # (n*h_unfold*w_unfold, c, unfold_mks, unfold_mks)
        # 'mks' is short for 'mask_kernel_size'
        context_cols = context_cols.reshape(-1, *context_cols.shape[2:])

        # the shape of correlation map should be:
        # (n, h_unfold*w_unfold, h', w')
        correlation_map = self.patch_correlation(x, context_cols)
        # fuse correlation map to enlarge consistent attention region.
        if self.with_fuse_correlation:
            correlation_map = self.fuse_correlation_map(
                correlation_map, h_unfold, w_unfold)

        correlation_map = self.mask_correlation_map(correlation_map, mask=mask)

        attention_score = self.softmax(correlation_map * self.softmax_scale)

        raw_context_filter = raw_context_cols.reshape(
            -1, *raw_context_cols.shape[2:])
        output = self.patch_copy_deconv(attention_score, raw_context_filter)
        # deconv will cause overlap and we need to remove the effects of that
        overlap_factor = self.calculate_overlap_factor(attention_score)
        output /= overlap_factor

        if self.return_attention_score:
            n, _, h_s, w_s = attention_score.size()
            attention_score = attention_score.view(n, h_unfold, w_unfold, h_s,
                                                   w_s)
            return output, attention_score

        return output

    def patch_correlation(self, x, kernel):
        """Calculate patch correlation.

        Args:
            x (torch.Tensor): Input tensor.
            kernel (torch.Tensor): Kernel tensor.

        Returns:
            torch.Tensor: Tensor with shape of (n, l, h, w).
        """
        n, _, h_in, w_in = x.size()

        patch_corr = F.conv2d(
            x.view(1, -1, h_in, w_in),
            kernel,
            stride=self.unfold_corr_stride,
            padding=self.unfold_corr_padding,
            dilation=self.unfold_corr_dilation,
            groups=n)
        h_out, w_out = patch_corr.size()[-2:]
        return patch_corr.view(n, -1, h_out, w_out)

    def patch_copy_deconv(self, attention_score, context_filter):
        """Copy patches using deconv.

        Args:
            attention_score (torch.Tensor): Tensor with shape of (n, l , h, w).
            context_filter (torch.Tensor): Filter kernel.

        Returns:
            torch.Tensor: Tensor with shape of (n, c, h, w).
        """
        n, num_context, h, w = attention_score.size()
        attention_score = attention_score.view(1, -1, h, w)
        output = F.conv_transpose2d(
            attention_score,
            context_filter,
            stride=self.unfold_raw_stride,
            padding=self.unfold_raw_padding,
            groups=n)
        h_out, w_out = output.size()[-2:]
        return output.view(n, -1, h_out, w_out)

    def fuse_correlation_map(self, correlation_map, h_unfold, w_unfold):
        """Fuse correlation map.

        This operation is to fuse correlation map for increasing large
        consistent correlation regions.

        The mechanism behind this op is simple and easy to understand. A
        standard 'Eye' matrix will be applied as a filter on the correlation
        map in horizontal and vertical direction.

        The shape of input correlation map is (n, h_unfold*w_unfold, h, w).
        When adopting fusing, we will apply convolutional filter in the
        reshaped feature map with shape of (n, 1, h_unfold*w_fold, h*w).

        A simple specification for horizontal direction is shown below:

        .. code-block:: python

                   (h, (h, (h, (h,
                    0)  1)  2)  3)  ...
            (h, 0)
            (h, 1)      1
            (h, 2)          1
            (h, 3)              1
            ...

        """
        # horizontal direction
        n, _, h_map, w_map = correlation_map.size()
        map_ = correlation_map.permute(0, 2, 3, 1)
        map_ = map_.reshape(n, h_map * w_map, h_unfold * w_unfold, 1)
        map_ = map_.permute(0, 3, 1, 2).contiguous()
        map_ = self.fuse_conv(map_, self.fuse_kernel)

        correlation_map = map_.view(n, h_unfold, w_unfold, h_map, w_map)

        # vertical direction
        map_ = correlation_map.permute(0, 2, 1, 4,
                                       3).reshape(n, 1, h_unfold * w_unfold,
                                                  h_map * w_map)
        map_ = self.fuse_conv(map_, self.fuse_kernel)

        # Note that the dimension should be transposed since the convolution of
        # eye matrix will put the normed scores into the last several dimension
        correlation_map = map_.view(n, w_unfold, h_unfold, w_map,
                                    h_map).permute(0, 4, 3, 2, 1)
        correlation_map = correlation_map.reshape(n, -1, h_unfold, w_unfold)

        return correlation_map

    def calculate_unfold_hw(self,
                            input_size,
                            kernel_size=3,
                            stride=1,
                            dilation=1,
                            padding=0):
        """Calculate (h, w) after unfolding

        The official implementation of `unfold` in pytorch will put the
        dimension (h, w) into `L`. Thus, this function is just to calculate the
        (h, w) according to the equation in:
        https://pytorch.org/docs/stable/nn.html#torch.nn.Unfold
        """
        h_in, w_in = input_size

        h_unfold = int((h_in + 2 * padding - dilation *
                        (kernel_size - 1) - 1) / stride + 1)

        w_unfold = int((w_in + 2 * padding - dilation *
                        (kernel_size - 1) - 1) / stride + 1)
        return h_unfold, w_unfold

    def calculate_overlap_factor(self, attention_score):
        """Calculte the overlap factor after applying deconv.

        Args:
            attention_score (torch.Tensor): The attention score with shape of
                (n, c, h, w).

        Returns:
            torch.Tensor: The overlap factor will be returned.
        """
        h, w = attention_score.shape[-2:]
        kernel_size = self.unfold_raw_kernel_size

        ones_input = torch.ones(1, 1, h, w).to(attention_score)
        ones_filter = torch.ones(1, 1, kernel_size,
                                 kernel_size).to(attention_score)
        overlap = F.conv_transpose2d(
            ones_input,
            ones_filter,
            stride=self.unfold_raw_stride,
            padding=self.unfold_raw_padding)

        # avoid division by zero
        overlap[overlap == 0] = 1.
        return overlap

    def mask_correlation_map(self, correlation_map, mask):
        """Add mask weight for correlation map.

        Add a negative infinity number to the masked regions so that softmax
        function will result in 'zero' in those regions.

        Args:
            correlation_map (torch.Tensor): Correlation map with shape of
                (n, h_unfold*w_unfold, h_map, w_map).
            mask (torch.Tensor): Mask tensor with shape of (n, c, h, w). '1'
                in the mask indicates masked region while '0' indicates valid
                region.

        Returns:
            torch.Tensor: Updated correlation map with mask.
        """
        if mask is not None:
            mask = F.interpolate(mask, scale_factor=self.scale)
            # if any pixel is masked in patch, the patch is considered to be
            # masked
            mask_cols = self.im2col(
                mask,
                kernel_size=self.unfold_corr_kernel_size,
                stride=self.unfold_corr_stride,
                padding=self.unfold_corr_padding,
                dilation=self.unfold_corr_dilation)
            mask_cols = (mask_cols.sum(dim=1, keepdim=True) > 0).float()
            mask_cols = mask_cols.permute(0, 2,
                                          1).reshape(mask.size(0), -1, 1, 1)
            # add negative inf will bring zero in softmax
            mask_cols[mask_cols == 1] = -float('inf')
            correlation_map += mask_cols
        return correlation_map

    def im2col(self,
               img,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               normalize=False,
               return_cols=False):
        """Reshape image-style feature to columns.

        This function is used for unfold feature maps to columns. The
        details of this function can be found in:
        https://pytorch.org/docs/1.1.0/nn.html?highlight=unfold#torch.nn.Unfold

        Args:
            img (torch.Tensor): Features to be unfolded. The shape of this
                feature should be (n, c, h, w).
            kernel_size (int): In this function, we only support square kernel
                with same height and width.
            stride (int): Stride number in unfolding. Default: 1.
            padding (int): Padding number in unfolding. Default: 0.
            dilation (int): Dilation number in unfolding. Default: 1.
            normalize (bool): If True, the unfolded feature will be normalized.
                Default: False.
            return_cols (bool): The official implementation in PyTorch of
                unfolding will return features with shape of
                (n, c*$prod{kernel_size}$, L). If True, the features will be
                reshaped to (n, L, c, kernel_size, kernel_size). Otherwise,
                the results will maintain the shape as the official
                implementation.

        Returns:
            torch.Tensor: Unfolded columns. If `return_cols` is True, the \
                shape of output tensor is \
                `(n, L, c, kernel_size, kernel_size)`. Otherwise, the shape \
                will be `(n, c*$prod{kernel_size}$, L)`.
        """

        # unfold img to columns with shape (n, c*kernel_size**2, num_cols)
        img_unfold = F.unfold(
            img,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)
        # normalize the feature map
        if normalize:
            norm = torch.sqrt((img_unfold**2).sum(dim=1, keepdim=True))
            eps = torch.tensor([1e-4]).to(img)
            img_unfold = img_unfold / torch.max(norm, eps)

        if return_cols:
            img_unfold_ = img_unfold.permute(0, 2, 1)
            n, num_cols = img_unfold_.size()[:2]
            img_cols = img_unfold_.view(n, num_cols, img.size(1), kernel_size,
                                        kernel_size)
            return img_cols

        return img_unfold
