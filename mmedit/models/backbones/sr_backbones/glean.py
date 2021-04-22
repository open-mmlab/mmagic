import math

import torch
import torch.nn as nn

from mmedit.models.backbones.sr_backbones.rrdb_net import RRDB
from mmedit.models.common import PixelShufflePack, make_layer
from mmedit.models.components.stylegan2.generator_discriminator import \
    StyleGANv2Generator
from mmedit.models.registry import BACKBONES

# import torch.nn.functional as F

# from mmcv.cnn import ConvModule

# from mmedit.utils import get_root_logger

# from mmcv.runner import load_checkpoint


@BACKBONES.register_module()
class GLEAN(StyleGANv2Generator):
    r"""GLEAN architecture for super-resolution.

    This method makes use of StyleGAN2 and hence the arguments follow that in
    'StyleGAN2v2Generator'.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of covolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered offical weights as
    follows:

    - styelgan2-ffhq-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - styelgan2-cat-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        generator = StyleGANv2Generator(1024, 512,
                                        pretrained=dict(
                                            ckpt_path=ckpt_http,
                                            prefix='generator_ema'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path. If you just want to load the original
    generator (not the ema model), please set the prefix with 'generator'.

    Note that our implementation allows to generate BGR image, while the
    original StyleGAN2 outputs RGB images by default. Thus, we provide
    ``bgr2rgb`` argument to convert the image space.

    Args:
        in_size (int): The size of the input image.
        out_size (int): The output size of the StyleGAN2 generator.
        style_channels (int): The number of channels for style code.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        channel_multiplier (int, optional): The mulitiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        lr_mlp (float, optional): The learning rate for the style mapping
            layer. Defaults to 0.01.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we defaultly adopt mixing style mode. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probabilty. The value should be
            in range of [0, 1]. Defaults to 0.9.
        pretrained (dict | None, optional): Information for pretained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
        bgr2rgb (bool, optional): Whether to flip the image channel dimension.
            Defaults to False.
    """

    def __init__(self,
                 in_size,
                 out_size,
                 style_channels,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 pretrained=None,
                 bgr2rgb=False):

        super().__init__(out_size, style_channels, num_mlps,
                         channel_multiplier, blur_kernel, lr_mlp,
                         default_style_mode, eval_style_mode, mix_prob,
                         pretrained, bgr2rgb)

        self.in_size = in_size

        # all the StyleGAN modules are fixed and set to eval mode
        self.requires_grad_(False)
        self.eval()

        # encoder
        num_styles = int(math.log2(out_size)) * 2 - 2
        encoder_res = [2**i for i in range(int(math.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(3, 64, num_blocks=23),
                nn.Conv2d(64, self.channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for res in encoder_res:
            in_channels = self.channels[res]
            if res > 4:
                out_channels = self.channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder.append(block)

        # additional modules for StyleGAN
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = self.channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
            self.fusion_skip.append(
                nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [
            2**i for i in range(
                int(math.log2(in_size)), int(math.log2(out_size) + 1))
        ]
        self.decoder = nn.ModuleList()
        for res in decoder_res:
            if res == in_size:
                in_channels = self.channels[res]
            else:
                in_channels = 2 * self.channels[res]

            if res < out_size:
                out_channels = self.channels[res * 2]
                self.decoder.append(
                    PixelShufflePack(
                        in_channels, out_channels, 2, upsample_kernel=3))
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, 3, 3, 1, 1)))

    def forward(self, lr):
        """Forward function.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN2, you can provide noise tensor or latent tensor. Given
                a list containing more than one noise or latent tensors, style
                mixing trick will be used in training. Of course, You can
                directly give a batch of noise through a ``torhc.Tensor`` or
                offer a callable function to sample a batch of noise data.
                Otherwise, the ``None`` indicates to use the default noise
                sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            inject_index (int | None, optional): The index number for mixing
                style codes. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncatioin trick will be adopted. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            injected_noise (torch.Tensor | None, optional): Given a tensor, the
                random noise will be fixed as this input injected noise.
                Defaults to None.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True.

        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary
                containing more data.
        """

        injected_noise = [
            getattr(self, f'injected_noise_{i}')
            for i in range(self.num_injected_noises)
        ]

        # encoder
        feat = lr
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.insert(0, feat)

        latent = encoder_features[0].view(lr.size(0), -1, self.style_channels)
        encoder_features = encoder_features[1:]

        # 4x4 stage
        out = self.constant_input(latent)
        out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        generator_features = []
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], injected_noise[1::2],
                injected_noise[2::2], self.to_rgbs):

            if out.size(2) <= self.in_size:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]

                out = torch.cat([out, feat], dim=1)
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat], dim=1)
                skip = self.fusion_skip[fusion_index](skip)

            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)

            if out.size(2) > self.in_size:
                generator_features.append(out)

            _index += 2

        # decoder
        hr = encoder_features[-1]
        for i, block in enumerate(self.decoder):
            if i > 0:
                hr = torch.cat([hr, generator_features[i - 1]], dim=1)
            hr = block(hr)

        return hr


class RRDBFeatureExtractor(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Defaults: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32):

        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            RRDB,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        return feat
