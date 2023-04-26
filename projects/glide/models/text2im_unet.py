import torch
import torch.nn as nn
import torch.nn.functional as F

from mmagic.models import DenoisingUnet
from mmagic.registry import MODELS
from .glide_modules import Transformer
from .glide_tokenizer import get_encoder


@MODELS.register_module()
class Text2ImUNet(DenoisingUnet):
    """A UNetModel used in GLIDE that conditions on text with an encoding
    transformer. Expects an extra kwarg `tokens` of text.

    Args:
        text_ctx (int): Number of text tokens to expect.
        xf_width (int): Width of the transformer.
        xf_layers (int): Depth of the transformer.
        xf_heads (int): Number of heads in the transformer.
        xf_final_ln (bool): Whether to use a LayerNorm after the output layer.
        tokenizer (callable, optional): Text tokenizer for sampling/vocab
            size. Defaults to get_encoder().
        cache_text_emb (bool, optional): Whether to cache text embeddings.
            Defaults to False.
        xf_ar (float, optional): Autoregressive weight for the transformer.
            Defaults to 0.0.
        xf_padding (bool, optional): Whether to use padding in the transformer.
            Defaults to False.
        share_unemb (bool, optional): Whether to share UNet embeddings.
            Defaults to False.
    """

    def __init__(
        self,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        *args,
        tokenizer=get_encoder(),
        cache_text_emb=False,
        xf_ar=0.0,
        xf_padding=False,
        share_unemb=False,
        **kwargs,
    ):
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_ar = xf_ar
        self.xf_padding = xf_padding
        self.tokenizer = tokenizer

        if not xf_width:
            super().__init__(*args, **kwargs, encoder_channels=None)
        else:
            super().__init__(*args, **kwargs, encoder_channels=xf_width)

        if self.xf_width:
            self.transformer = Transformer(
                text_ctx,
                xf_width,
                xf_layers,
                xf_heads,
            )
            if xf_final_ln:
                self.final_ln = nn.LayerNorm(xf_width)
            else:
                self.final_ln = None

            self.token_embedding = nn.Embedding(self.tokenizer.n_vocab,
                                                xf_width)
            self.positional_embedding = nn.Parameter(
                torch.empty(text_ctx, xf_width, dtype=torch.float32))
            self.transformer_proj = nn.Linear(xf_width, self.base_channels * 4)

            if self.xf_padding:
                self.padding_embedding = nn.Parameter(
                    torch.empty(text_ctx, xf_width, dtype=torch.float32))
            if self.xf_ar:
                self.unemb = nn.Linear(xf_width, self.tokenizer.n_vocab)
                if share_unemb:
                    self.unemb.weight = self.token_embedding.weight

        self.cache_text_emb = cache_text_emb
        self.cache = None

    def get_text_emb(self, tokens, mask):
        assert tokens is not None

        if self.cache_text_emb and self.cache is not None:
            assert (tokens == self.cache['tokens']).all(
            ), f"Tokens {tokens.cpu().numpy().tolist()} do not match \
            cache {self.cache['tokens'].cpu().numpy().tolist()}"

            return self.cache

        xf_in = self.token_embedding(tokens.long())
        xf_in = xf_in + self.positional_embedding[None]
        if self.xf_padding:
            assert mask is not None
            xf_in = torch.where(mask[..., None], xf_in,
                                self.padding_embedding[None])
        xf_out = self.transformer(xf_in.to(self.dtype))
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        if self.cache_text_emb:
            self.cache = dict(
                tokens=tokens,
                xf_proj=xf_proj.detach(),
                xf_out=xf_out.detach() if xf_out is not None else None,
            )

        return outputs

    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps, tokens=None, mask=None):
        hs = []
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps],
                                     dtype=torch.long,
                                     device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        if timesteps.shape[0] != x.shape[0]:
            timesteps = timesteps.repeat(x.shape[0])
        emb = self.time_embedding(timesteps)
        if self.xf_width:
            text_outputs = self.get_text_emb(tokens, mask)
            xf_proj, xf_out = text_outputs['xf_proj'], text_outputs['xf_out']
            emb = emb + xf_proj.to(emb)
        else:
            xf_out = None
        h = x.type(self.dtype)
        for module in self.in_blocks:
            h = module(h, emb, xf_out)
            hs.append(h)
        h = self.mid_blocks(h, emb, xf_out)
        for module in self.out_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h


@MODELS.register_module()
class SuperResText2ImUNet(Text2ImUNet):
    """A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if 'in_channels' in kwargs:
            kwargs = dict(kwargs)
            kwargs['in_channels'] = kwargs['in_channels'] * 2
        else:
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width),
            mode='bilinear',
            align_corners=False)
        x = torch.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)
