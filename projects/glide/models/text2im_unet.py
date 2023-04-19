import torch
import torch.nn as nn

from mmagic.models import DenoisingUnet
from mmagic.registry import MODELS
from .glide_modules import Transformer
from .glide_tokenizer import get_encoder


@MODELS.register_module()
class Text2ImUNet(DenoisingUnet):
    """A UNetModel that conditions on text with an encoding transformer.
    Expects an extra kwarg `tokens` of text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param tokenizer: the text tokenizer for sampling/vocab size.
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

        # del self.label_embedding

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

    # def convert_to_fp16(self):
    #     super().convert_to_fp16()
    #     if self.xf_width:
    #         self.transformer.apply(convert_module_to_f16)
    #         self.transformer_proj.to(torch.float16)
    #         self.token_embedding.to(torch.float16)
    #         self.positional_embedding.to(torch.float16)
    #         if self.xf_padding:
    #             self.padding_embedding.to(torch.float16)
    #         if self.xf_ar:
    #             self.unemb.to(torch.float16)

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

        # TODO not sure
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
