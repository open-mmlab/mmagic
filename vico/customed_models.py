from typing import Optional, Dict, Any, Tuple, Union
from diffusers import UNet2DConditionModel, Transformer2DModel, AutoencoderKL
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn, ResnetBlock2D, DownBlock2D, UpBlock2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from diffusers.utils import is_torch_version
from diffusers.models.unet_2d_condition import UNet2DConditionOutput


def otsu(mask_in):
    # normalize
    mask_norm = (mask_in - mask_in.min(-1, keepdim=True)[0]) / \
       (mask_in.max(-1, keepdim=True)[0] - mask_in.min(-1, keepdim=True)[0])
    
    bs = mask_in.shape[0]
    h = mask_in.shape[1]
    mask = []
    for i in range(bs):
        threshold_t = 0.
        max_g = 0.
        for t in range(10):
            mask_i = mask_norm[i]
            low = mask_i[mask_i < t*0.1]
            high = mask_i[mask_i >= t*0.1]
            low_num = low.shape[0]/h
            high_num = high.shape[0]/h
            low_mean = low.mean()
            high_mean = high.mean()
        
            g = low_num*high_num*((low_mean-high_mean)**2)
            if g > max_g:
                max_g = g
                threshold_t = t*0.1
            
        mask_i[mask_i < threshold_t] = 0
        mask_i[mask_i > threshold_t] = 1
        mask.append(mask_i)
    mask_out = torch.stack(mask, dim=0)
            
    return mask_out


@dataclass
class ViCoTransformer2DModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            Hidden states conditioned on `encoder_hidden_states` input. If discrete, returns probability distributions
            for the unnoised latent pixels.
    """

    sample: torch.FloatTensor
    loss_reg: torch.FloatTensor


class ViCoTransformer2D(nn.Module):

    def __init__(self, org_transformer2d: Transformer2DModel, cross_attention_dim) -> None:
        super().__init__()
        self.transformer_blocks = org_transformer2d.transformer_blocks
        self.is_input_continuous = org_transformer2d.is_input_continuous
        self.norm = org_transformer2d.norm
        self.use_linear_projection = org_transformer2d.use_linear_projection
        self.proj_in = org_transformer2d.proj_in
        self.proj_out = org_transformer2d.proj_out
        self.is_input_vectorized = org_transformer2d.is_input_vectorized
        self.is_input_patches = org_transformer2d.is_input_patches

        num_attention_heads = org_transformer2d.num_attention_heads
        attention_head_dim = org_transformer2d.attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.image_cross_attention = BasicTransformerBlock(
            inner_dim, num_attention_heads, attention_head_dim, cross_attention_dim=cross_attention_dim)
        self.image_cross_attention.to(org_transformer2d.device, dtype=org_transformer2d.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        placeholder_position: list = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            encoder_attention_mask ( `torch.Tensor`, *optional* ).
                Cross-attention mask, applied to encoder_hidden_states. Two formats supported:
                    Mask `(batch, sequence_length)` True = keep, False = discard. Bias `(batch, 1, sequence_length)` 0
                    = keep, -10000 = discard.
                If ndim == 2: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            hidden_states, image_reference = hidden_states.chunk(2, dim=0)
            batch, _, height, width = hidden_states.shape
            residual = hidden_states
            image_reference_residual = image_reference

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = self.proj_in(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
            image_reference = block(
                image_reference,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
            attention_probs = block.attn2.attn_probs

        # 2.5. image cross attention
        ph_idx, eot_idx = placeholder_position[0], placeholder_position[1]
        attn = attention_probs.transpose(1,2)
        attn_ph = attn[ph_idx].squeeze(1) # bs, n_patch
        attn_eot = attn[eot_idx].squeeze(1).detach()
        
        # ########################
        # attention reg
        if self.image_cross_attention.training:
            loss_reg = F.mse_loss(attn_ph/attn_ph.max(-1, keepdim=True)[0], attn_eot/attn_eot.max(-1, keepdim=True)[0])
        # ########################
            
        mask = attn_ph.detach()
        mask = otsu(mask).bool()
        
        hidden_states = self.image_cross_attention(hidden_states, context=image_reference, mask=mask)

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = self.proj_out(hidden_states)

                image_reference = image_reference.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                image_reference = self.proj_out(image_reference)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

                image_reference = image_reference.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                image_reference = self.proj_out(image_reference)

            output = hidden_states + residual
            image_reference = image_reference + image_reference_residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()
        elif self.is_input_patches:
            # TODO: cleanup!
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)

            # unpatchify
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        output = torch.cat([output, image_reference], dim=0)
        if not return_dict:
            return (output, loss_reg)

        return ViCoTransformer2DModelOutput(sample=output, loss_reg=loss_reg)


def replace_transformer2d(module: nn.Module, cross_attention_dim):

    transformer2d_modules = [(k.rsplit(".", 1), v) for k, v in module.named_modules() if isinstance(v, Transformer2DModel)]
    for (parent, k), v in transformer2d_modules:
        parent = module.get_submodule(parent)
        setattr(parent, k, ViCoTransformer2D(v, cross_attention_dim))


class ViCoModuleWrapper(nn.Module):
    def __init__(self, org_module) -> None:
        super().__init__()
        self.org_module = org_module
        self.org_module.forward = self.forward


class ViCoCrossAttnDownBlock2D(ViCoModuleWrapper):

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        placeholder_position: torch.Tensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        output_states = ()
        loss_reg_all = 0.0

        for resnet, attn in zip(self.org_module.resnets, self.org_module.attentions):
            attn: ViCoTransformer2D
            if self.org_module.training and self.org_module.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states, loss_reg = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    placeholder_position,
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states, loss_reg = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    placeholder_position=placeholder_position,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

            output_states = output_states + (hidden_states,)
            loss_reg_all += loss_reg

        if self.org_module.downsamplers is not None:
            for downsampler in self.org_module.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states, loss_reg_all
    

class ViCoUNetMidBlock2DCrossAttn(ViCoModuleWrapper):

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        placeholder_position: torch.Tensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        loss_reg_all = 0.0
        hidden_states = self.org_module.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.org_module.attentions, self.org_module.resnets[1:]):
            hidden_states, loss_reg = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                placeholder_position=placeholder_position,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            hidden_states = resnet(hidden_states, temb)
            loss_reg_all += loss_reg

        return hidden_states, loss_reg_all

class ViCoCrossAttnUpBlock2D(ViCoModuleWrapper):

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        placeholder_position: list = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        loss_reg_all = 0.0
        for resnet, attn in zip(self.org_module.resnets, self.org_module.attentions):
            attn: ViCoTransformer2D

            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.org_module.training and self.org_module.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states, loss_reg = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    placeholder_position,
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states, loss_reg = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    placeholder_position=placeholder_position,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            loss_reg_all += loss_reg

        if self.org_module.upsamplers is not None:
            for upsampler in self.org_module.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states, loss_reg_all


class ViCoUNet2DConditionOutput(UNet2DConditionOutput):
    sample: torch.FloatTensor
    loss_reg: torch.FloatTensor


class ViCoUNet2DConditionModel(ViCoModuleWrapper):

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        placeholder_position: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            encoder_attention_mask (`torch.Tensor`):
                (batch, sequence_length) cross-attention mask, applied to encoder_hidden_states. True = keep, False =
                discard. Mask will be converted into a bias, which adds large negative values to attention scores
                corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            added_cond_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified includes additonal conditions that can be used for additonal time
                embeddings or encoder hidden states projections. See the configurations `encoder_hid_dim_type` and
                `addition_embed_type` for more information.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.org_module.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.org_module.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.org_module.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.org_module.time_embedding(t_emb, timestep_cond)

        if self.org_module.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.org_module.config.class_embed_type == "timestep":
                class_labels = self.org_module.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.org_module.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.org_module.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.org_module.config.addition_embed_type == "text":
            aug_emb = self.org_module.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb
        elif self.org_module.config.addition_embed_type == "text_image":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.org_module.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)

            aug_emb = self.org_module.add_embedding(text_embs, image_embs)
            emb = emb + aug_emb

        if self.org_module.time_embed_act is not None:
            emb = self.org_module.time_embed_act(emb)

        if self.org_module.encoder_hid_proj is not None and self.org_module.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.org_module.encoder_hid_proj(encoder_hidden_states)
        elif self.org_module.encoder_hid_proj is not None and self.org_module.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.org_module.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.org_module.encoder_hid_proj(encoder_hidden_states, image_embeds)

        # 2. pre-process
        sample = self.org_module.conv_in(sample)

        loss_reg_all = 0.0

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.org_module.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples, loss_reg = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    placeholder_position=placeholder_position,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
                loss_reg_all += loss_reg
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.org_module.mid_block is not None:
            sample = self.org_module.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.org_module.up_blocks):
            is_final_block = i == len(self.org_module.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample, loss_reg = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    placeholder_position=placeholder_position,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                loss_reg_all += loss_reg
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.org_module.conv_norm_out:
            sample = self.org_module.conv_norm_out(sample)
            sample = self.org_module.conv_act(sample)
        sample = self.org_module.conv_out(sample)

        if not return_dict:
            return (sample, loss_reg_all)

        return UNet2DConditionOutput(sample=sample, loss_reg=loss_reg_all)


def set_vico_modules(unet):
    replace_transformer2d(unet, cross_attention_dim=unet.config.cross_attention_dim)
    vico_unet = ViCoUNet2DConditionModel(unet)
    for name, layer in unet.named_modules():
        if layer.__class__.__name__ in ("CrossAttnDownBlock2D", "CrossAttnUpBlock2D", "UNetMidBlock2DCrossAttn"):
            new_module = eval("ViCo" + layer.__class__.__name__)(layer)
