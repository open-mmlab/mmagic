# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Any, Callable, Dict, Tuple, Type

import torch


def add_tome_cfg_hook(model: torch.nn.Module):
    """Add a forward pre hook to get the image size. This hook can be removed
    with remove_patch.

    Source: https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L158 # noqa
    """

    def hook(module, args):
        module._tome_info['size'] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info['hooks'].append(model.register_forward_pre_hook(hook))


def build_mmagic_wrapper_tomesd_block(block_class: Type[torch.nn.Module]
                                      ) -> Type[torch.nn.Module]:
    """Make a patched class for a DiffusersWrapper model in mmagic. This patch
    applies ToMe to the forward function of the block.

    Refer to: https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L67 # noqa
    Args:
        block_class (torch.nn.Module): original class need tome speedup.

    Returns:
        ToMeBlock (torch.nn.Module): patched class based on the original class.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            timestep=None,
            attention_mask=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ):
            # -> (1) ToMeBlock
            m_a, m_c, m_m, u_a, u_c, u_m = build_merge(hidden_states,
                                                       self._tome_info)
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa,\
                    shift_mlp, scale_mlp, gate_mlp = self.norm1(
                                    hidden_states,
                                    timestep,
                                    class_labels,
                                    hidden_dtype=hidden_states.dtype)
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # -> (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            if cross_attention_kwargs is not None:
                cross_attention_kwargs = cross_attention_kwargs
            else:
                cross_attention_kwargs = {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states
                if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # -> (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm else self.norm2(hidden_states))
                # -> (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                # -> (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (
                    1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # -> (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # -> (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock


def build_mmagic_tomesd_block(block_class: Type[torch.nn.Module]
                              ) -> Type[torch.nn.Module]:
    """Make a patched class for a mmagic StableDiffusion model. This patch
    applies ToMe to the forward function of the block.

    Refer to: https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L67 # noqa
    Args:
        block_class (torch.nn.Module): original class need tome speedup.

    Returns:
        ToMeBlock (torch.nn.Module): patched class based on the original class.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(self, hidden_states, context=None, timestep=None):
            # ->(1) ToMeBlock
            m_a, m_c, m_m, u_a, u_c, u_m = build_merge(hidden_states,
                                                       self._tome_info)

            # 1. Self-Attention
            # ->(2) ToMe m_a
            norm_hidden_states = (m_a(self.norm1(hidden_states)))

            # ->(3) ToMe u_a
            if self.only_cross_attention:
                hidden_states = u_a(self.attn1(norm_hidden_states,
                                               context)) + hidden_states
            else:
                hidden_states = u_a(
                    self.attn1(norm_hidden_states)) + hidden_states

            # 2. Cross-Attention
            # ->(4) ToMe m_c
            norm_hidden_states = (m_c(self.norm2(hidden_states)))
            # ->(5) ToMe u_c
            hidden_states = u_c(
                self.attn2(norm_hidden_states,
                           context=context)) + hidden_states

            # 3. Feed-forward
            # ->(6) ToMe m_m, u_m
            hidden_states = u_m(self.ff(m_m(
                self.norm3(hidden_states)))) + hidden_states

            return hidden_states

    return ToMeBlock


def isinstance_str(x: object, cls_name: str):
    """Checks whether `x` has any class *named* `cls_name` in its ancestry.
    Doesn't require access to the class's implementation.

    Source: https://github.com/dbolya/tomesd/blob/main/tomesd/utils.py#L3 # noqa
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


def do_nothing(x: torch.Tensor, mode: str = None):
    """Build identical mapping function.

    Source: https://github.com/dbolya/tomesd/blob/main/tomesd/merge.py#L5 # noqa
    """
    return x


def mps_gather_workaround(input, dim, index):
    """Gather function specific for `mps` backend (Metal Performance Shaders).

    Source: https://github.com/dbolya/tomesd/blob/main/tomesd/merge.py#L9 # noqa
    """
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1), dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int,
                                     h: int,
                                     sx: int,
                                     sy: int,
                                     r: int,
                                     no_rand: bool = False
                                     ) -> Tuple[Callable, Callable]:
    """Partitions the tokens into src and dst and merges r tokens from src to
    dst, dst tokens are partitioned by choosing one randomy in each (`sx`,
    `sy`) region. More details refer to `Token Merging: Your ViT But Faster`,
    paper link: <https://arxiv.org/abs/2210.09461>`_ # noqa.

    Source: https://github.com/dbolya/tomesd/blob/main/tomesd/merge.py#20 # noqa

    Args:
        metric (torch.Tensor): metric with size (B, N, C) for similarity computation.
        w (int): image width in tokens.
        h (int): image height in tokens.
        sx (int): stride in the x dimension for dst, must divide `w`.
        sy (int): stride in the y dimension for dst, must divide `h`.
        r (int): number of tokens to remove (by merging).
        no_rand (bool): if true, disable randomness (use top left corner only).

    Returns:
        merge (Callable): token merging function.
        unmerge (Callable): token unmerging function.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    if metric.device.type == 'mps':
        gather = mps_gather_workaround
    else:
        gather = torch.gather

    with torch.no_grad():

        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to
        # be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(
                hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(
                sy * sx, size=(hsy, wsx, 1), device=metric.device)

        # The image might not divide sx and sy, so we need to work
        # on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(
            hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(
            dim=2,
            index=rand_idx,
            src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(
            1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it
        # into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(
                h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort
        # gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode='mean') -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))

        if not hasattr(torch.Tensor,
                       'scatter_reduce') or torch.__version__ < '1.12.1':
            raise ImportError(
                'Please upgrade torch >= 1.12.1 to enable \'scatter_reduce\'')
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1,
                index=unm_idx).expand(B, unm_len, c),
            src=unm)
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1,
                index=src_idx).expand(B, r, c),
            src=src)

        return out

    return merge, unmerge


def build_merge(x: torch.Tensor, tome_info: Dict[str,
                                                 Any]) -> Tuple[Callable, ...]:
    """Build the merge and unmerge functions for a given setting from
    `tome_info`.

    Source: https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L10 # noqa
    """
    original_h, original_w = tome_info['size']
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info['args']

    if downsample <= args['max_downsample']:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args['ratio'])
        # If the batch size is odd, then it's not possible for promted and
        # unprompted images to be in the same batch, which causes artifacts
        # with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args['use_rand']
        m, u = bipartite_soft_matching_random2d(x, w, h, args['sx'],
                                                args['sy'], r, not use_rand)
    else:
        m, u = (do_nothing, do_nothing)

    m_a, u_a = (m, u) if args['merge_attn'] else (do_nothing, do_nothing)
    m_c, u_c = (m, u) if args['merge_crossattn'] else (do_nothing, do_nothing)
    m_m, u_m = (m, u) if args['merge_mlp'] else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m
