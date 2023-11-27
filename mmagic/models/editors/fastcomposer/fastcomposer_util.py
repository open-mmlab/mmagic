# Copyright (c) OpenMMLab. All rights reserved.
import gc
import types
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.nn import Linear
from transformers import (CLIPModel, CLIPPreTrainedModel, CLIPTextModel,
                          CLIPVisionConfig, CLIPVisionModel)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from mmagic.utils import try_import

_expand_mask = try_import('transformers.models.clip.modeling_clip')
if _expand_mask is None:
    _expand_mask = try_import(
        'ransformers.models.clip.modeling_clip._prepare_4d_attention_mask')


class FastComposerModel(nn.Module):
    """FastComposerModel is based on the StableDiffusion Model and the Clip
    Model."""

    def __init__(self, text_encoder, image_encoder, vae, unet, cfg):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vae = vae
        self.unet = unet
        self.use_ema = False
        self.ema_param = None
        self.pretrained_model_name_or_path = cfg[
            'pretrained_model_name_or_path']
        self.revision = cfg['revision']
        self.non_ema_revision = cfg['non_ema_revision']
        self.object_localization = cfg['object_localization']
        self.object_localization_weight = cfg['object_localization_weight']
        self.localization_layers = cfg['localization_layers']
        self.mask_loss = cfg['mask_loss']
        self.mask_loss_prob = cfg['mask_loss_prob']

        embed_dim = text_encoder.config.hidden_size

        self.postfuse_module = FastComposerPostfuseModule(embed_dim)

        if self.object_localization:
            self.cross_attention_scores = {}
            self.unet = unet_store_cross_attention_scores(
                self.unet, self.cross_attention_scores,
                self.localization_layers)
            self.object_localization_loss_fn = BalancedL1Loss(
                cfg['object_localization_threshold'],
                cfg['object_localization_normalize'],
            )

    def _clear_cross_attention_scores(self):
        """Delete cross attention scores."""
        if hasattr(self, 'cross_attention_scores'):
            keys = list(self.cross_attention_scores.keys())
            for k in keys:
                del self.cross_attention_scores[k]

        gc.collect()

    @staticmethod
    def from_pretrained(cfg, vae, unet):
        """Init FastComposerTextEncoder and FastComposerCLIPImageEncoder."""

        text_encoder = FastComposerTextEncoder.from_pretrained(
            cfg['pretrained_model_name_or_path'],
            subfolder='text_encoder',
            revision=cfg['revision'],
        )
        if not isinstance(cfg['image_encoder'], dict):
            image_encoder = FastComposerCLIPImageEncoder.from_pretrained(
                cfg['image_encoder'])
        else:
            vision_model = CLIPVisionModel(
                CLIPVisionConfig.from_dict(cfg['image_encoder']))
            visual_projection = Linear(
                in_features=1024, out_features=768, bias=False)
            vision_processor = T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )
            image_encoder = FastComposerCLIPImageEncoder(
                vision_model,
                visual_projection,
                vision_processor,
            )

        return FastComposerModel(text_encoder, image_encoder, vae, unet, cfg)

    def forward(self, batch, noise_scheduler):
        """Forward function.

        Args:
            batch (torch.Tensor ):
                You can directly input a ``torch.Tensor``.
            noise_scheduler (torch.Tensor ):
                You can directly input a ``torch.Tensor``.

        Returns:
            Dict
        """

        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']
        image_token_mask = batch['image_token_mask']
        object_pixel_values = batch['object_pixel_values']
        num_objects = batch['num_objects']

        vae_dtype = self.vae.parameters().__next__().dtype
        vae_input = pixel_values.to(vae_dtype)

        latents = self.vae.encode(vae_input).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps, (bsz, ),
            device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to
        # the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # (bsz, max_num_objects, num_image_tokens, dim)
        object_embeds = self.image_encoder(object_pixel_values)

        encoder_hidden_states = self.text_encoder(
            input_ids, image_token_mask, object_embeds,
            num_objects)[0]  # (bsz, seq_len, dim)

        encoder_hidden_states = self.postfuse_module(
            encoder_hidden_states,
            object_embeds,
            image_token_mask,
            num_objects,
        )

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif noise_scheduler.config.prediction_type == 'v_prediction':
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError('Unknown prediction type '
                             f'{noise_scheduler.config.prediction_type}')

        pred = self.unet(noisy_latents, timesteps,
                         encoder_hidden_states).sample

        if self.mask_loss and torch.rand(1) < self.mask_loss_prob:
            object_segmaps = batch['object_segmaps']
            mask = (object_segmaps.sum(dim=1) > 0).float()
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(pred.shape[-2], pred.shape[-1]),
                mode='bilinear',
                align_corners=False,
            )
            pred = pred * mask
            target = target * mask

        denoise_loss = F.mse_loss(
            pred.float(), target.float(), reduction='mean')

        return_dict = {'denoise_loss': denoise_loss}

        if self.object_localization:
            object_segmaps = batch['object_segmaps']
            image_token_idx = batch['image_token_idx']
            image_token_idx_mask = batch['image_token_idx_mask']
            localization_loss = get_object_localization_loss(
                self.cross_attention_scores,
                object_segmaps,
                image_token_idx,
                image_token_idx_mask,
                self.object_localization_loss_fn,
            )
            return_dict['localization_loss'] = localization_loss
            loss = self.object_localization_weight * localization_loss
            loss += denoise_loss
            self._clear_cross_attention_scores()
        else:
            loss = denoise_loss

        return_dict['loss'] = loss
        return return_dict


class FastComposerTextEncoder(CLIPPreTrainedModel):
    """TextEncoder for FastComposerModel."""

    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        """Init textEncoder with Stable Diffusion Model name or path."""
        model = CLIPTextModel.from_pretrained(model_name_or_path, **kwargs)
        text_model = model.text_model
        return FastComposerTextEncoder(text_model)

    def __init__(self, text_model):
        super().__init__(text_model.config)
        self.config = text_model.config
        self.final_layer_norm = text_model.final_layer_norm
        self.embeddings = text_model.embeddings
        self.encoder = text_model.encoder
        self._build_causal_attention_mask = build_causal_attention_mask

    def forward(
        self,
        input_ids,
        image_token_mask=None,
        object_embeds=None,
        num_objects=None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """Forward function.

        Args:
            input_ids (torch.Tensor ):
                You can directly input a ``torch.Tensor``.
            image_token_mask (torch.Tensor ):
                You can directly input a ``torch.Tensor``.
            object_embeds (torch.Tensor ):
                You can directly input a ``torch.Tensor``.
            num_objects (torch.Tensor ):
                You can directly input a ``torch.Tensor``.
            attention_mask (torch.Tensor ):
                You can directly input a ``torch.Tensor``.
            output_attentions (bool ):
                Default to None.
            output_hidden_states (bool ):
                Default to None.
            return_dict (bool ):
                Default to None.

        Returns:
            Union[Tuple, BaseModelOutputWithPooling]
        """

        output_attentions = (
            output_attentions if output_attentions is not None else
            self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = (
            return_dict
            if return_dict is not None else self.config.use_return_dict)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids)

        bsz, seq_len = input_shape
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype).to(hidden_states.device)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding
        # (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility:
        # argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(
                last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device
                         ).argmax(dim=-1), ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FastComposerCLIPImageEncoder(CLIPPreTrainedModel):
    """CLIPImageEncoder for FastComposerModel."""

    @staticmethod
    def from_pretrained(global_model_name_or_path):
        """Init CLIPModel with Clip model name or path."""

        model = CLIPModel.from_pretrained(global_model_name_or_path)
        vision_model = model.vision_model
        visual_projection = model.visual_projection
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        return FastComposerCLIPImageEncoder(
            vision_model,
            visual_projection,
            vision_processor,
        )

    def __init__(
        self,
        vision_model,
        visual_projection,
        vision_processor,
    ):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = vision_model.config.image_size

    def forward(self, object_pixel_values):
        """Forward function.

        Args:
            object_pixel_values (torch.Tensor ):
                You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """

        b, num_objects, c, h, w = object_pixel_values.shape

        object_pixel_values = object_pixel_values.view(b * num_objects, c, h,
                                                       w)

        if h != self.image_size or w != self.image_size:
            h, w = self.image_size, self.image_size
            object_pixel_values = F.interpolate(
                object_pixel_values, (h, w), mode='bilinear')

        object_pixel_values = self.vision_processor(object_pixel_values)
        object_embeds = self.vision_model(object_pixel_values)[1]
        object_embeds = self.visual_projection(object_embeds)
        object_embeds = object_embeds.view(b, num_objects, 1, -1)
        return object_embeds


def get_object_transforms(cfg):
    """Get Object transforms."""

    if cfg['no_object_augmentation']:
        pre_augmentations = []
        augmentations = []
    else:
        pre_augmentations = [
            (
                'zoomin',
                T.RandomApply([RandomZoomIn(min_zoom=1.0, max_zoom=2.0)],
                              p=0.5),
            ),
        ]

        augmentations = [
            (
                'rotate',
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=30,
                            interpolation=T.InterpolationMode.BILINEAR)
                    ],
                    p=0.75,
                ),
            ),
            ('jitter', T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.5)],
                                     p=0.5)),
            ('blur', T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))],
                                   p=0.5)),
            ('gray', T.RandomGrayscale(p=0.1)),
            ('flip', T.RandomHorizontalFlip()),
            ('elastic', T.RandomApply([T.ElasticTransform()], p=0.5)),
        ]

    object_transforms = torch.nn.Sequential(
        OrderedDict([
            *pre_augmentations,
            ('pad_to_square', PadToSquare(fill=0, padding_mode='constant')),
            (
                'resize',
                T.Resize(
                    (cfg['object_resolution'], cfg['object_resolution']),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
            ),
            *augmentations,
            ('convert_to_float', T.ConvertImageDtype(torch.float32)),
        ]))
    return object_transforms


class FastComposerPostfuseModule(nn.Module):
    """Postfuse Module for FastComposerModel."""

    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(
            embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, text_embeds, object_embeds):
        """Fuse function.

        Args:
            text_embeds (torch.Tensor ):
                You can directly input a ``torch.Tensor``.
            object_embeds (torch.Tensor ):
                You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """

        text_object_embeds = torch.cat([text_embeds, object_embeds], dim=-1)
        text_object_embeds = self.mlp1(text_object_embeds) + text_embeds
        text_object_embeds = self.mlp2(text_object_embeds)
        text_object_embeds = self.layer_norm(text_object_embeds)
        return text_object_embeds

    def forward(
        self,
        text_embeds,
        object_embeds,
        image_token_mask,
        num_objects,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            text_embeds (torch.Tensor ):
                You can directly input a ``torch.Tensor``.
            object_embeds (torch.Tensor ):
                You can directly input a ``torch.Tensor``.
            image_token_mask (torch.Tensor ):
                You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """

        text_object_embeds = fuse_object_embeddings(text_embeds,
                                                    image_token_mask,
                                                    object_embeds, num_objects,
                                                    self.fuse_fn)

        return text_object_embeds


def unet_store_cross_attention_scores(unet, attention_scores, layers=5):
    """Unet store cross attention scores."""
    from diffusers.models.attention_processor import (Attention, AttnProcessor,
                                                      AttnProcessor2_0)

    UNET_LAYER_NAMES = [
        'down_blocks.0',
        'down_blocks.1',
        'down_blocks.2',
        'mid_block',
        'up_blocks.1',
        'up_blocks.2',
        'up_blocks.3',
    ]

    start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        """Wrapper Function of create attention scores for unet."""

        def new_get_attention_scores(module, query, key, attention_mask=None):
            """Create attention scores for unet."""
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask)
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and 'attn2' in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module)

    return unet


class BalancedL1Loss(nn.Module):
    """BalancedL1Loss for object localization."""

    def __init__(self, threshold=1.0, normalize=False):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def forward(self, object_token_attn_prob, object_segmaps):
        """Forward function.

        Args:
            object_token_attn_prob (torch.Tensor ):
                You can directly input a ``torch.Tensor``.
            object_segmaps (torch.Tensor ):
                You can directly input a ``torch.Tensor``.

        Returns:
            float : ``float`` will be returned.
        """
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5)
        background_segmaps = 1 - object_segmaps
        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(
            dim=2) / background_segmaps_sum

        object_loss = (object_token_attn_prob *
                       object_segmaps).sum(dim=2) / object_segmaps_sum

        return background_loss - object_loss


def get_object_localization_loss(
    cross_attention_scores,
    object_segmaps,
    image_token_idx,
    image_token_idx_mask,
    loss_fn,
):
    """To obtain the average of the loss for each layer of object
    localization."""

    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        layer_loss = get_object_localization_loss_for_one_layer(
            v, object_segmaps, image_token_idx, image_token_idx_mask, loss_fn)
        loss += layer_loss
    return loss / num_layers


def get_object_localization_loss_for_one_layer(
    cross_attention_scores,
    object_segmaps,
    object_token_idx,
    object_token_idx_mask,
    loss_fn,
):
    """Get object localization loss for one layer."""

    bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b, max_num_objects, _, _ = object_segmaps.shape
    size = int(num_noise_latents**0.5)

    # Resize the object segmentation maps to
    # the size of the cross attention scores
    object_segmaps = F.interpolate(
        object_segmaps, size=(size, size), mode='bilinear')
    # (b, max_num_objects, size, size)

    object_segmaps = object_segmaps.view(
        b, max_num_objects, -1)  # (b, max_num_objects, num_noise_latents)

    num_heads = bxh // b

    cross_attention_scores = cross_attention_scores.view(
        b, num_heads, num_noise_latents, num_text_tokens)

    # Gather object_token_attn_prob
    object_token_attn_prob = torch.gather(
        cross_attention_scores,
        dim=3,
        index=object_token_idx.view(b, 1, 1, max_num_objects).expand(
            b, num_heads, num_noise_latents, max_num_objects),
    )  # (b, num_heads, num_noise_latents, max_num_objects)

    object_segmaps = (
        object_segmaps.permute(0, 2,
                               1).unsqueeze(1).expand(b, num_heads,
                                                      num_noise_latents,
                                                      max_num_objects))

    loss = loss_fn(object_token_attn_prob, object_segmaps)

    loss = loss * object_token_idx_mask.view(b, 1, max_num_objects)
    object_token_cnt = object_token_idx_mask.sum(dim=1).view(b, 1) + 1e-5
    loss = (loss.sum(dim=2) / object_token_cnt).mean()

    return loss


class RandomZoomIn(nn.Module):
    """RandomZoomIn for object transform."""

    def __init__(self, min_zoom=1.0, max_zoom=1.5):
        super().__init__()
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    def forward(self, image: torch.Tensor):
        """Forward function.

        Args:
            image (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        zoom = torch.rand(1) * (self.max_zoom - self.min_zoom) + self.min_zoom
        image = T.functional.resize(
            image,
            (int(zoom * image.shape[1]), int(zoom * image.shape[2])),
            interpolation=T.InterpolationMode.BILINEAR,
        )
        # crop top square
        image = CropTopSquare()(image)
        return image


class PadToSquare(nn.Module):
    """If the height of the image is greater than the width, padding will be
    added on both sides of the image to make it a square."""

    def __init__(self, fill=0, padding_mode='constant'):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor):
        """Forward function.

        Args:
            image (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        _, h, w = image.shape
        if h == w:
            return image
        elif h > w:
            padding = (h - w) // 2
            image = torch.nn.functional.pad(
                image,
                (padding, padding, 0, 0),
                self.padding_mode,
                self.fill,
            )
        else:
            padding = (w - h) // 2
            image = torch.nn.functional.pad(
                image,
                (0, 0, padding, padding),
                self.padding_mode,
                self.fill,
            )
        return image


class CropTopSquare(nn.Module):
    """If the height of the image is greater than the width, the image will be
    cropped into a square starting from the top of the image."""

    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        """Forward function.

        Args:
            image (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        _, h, w = image.shape
        if h <= w:
            return image
        return image[:, :w, :]


class MLP(nn.Module):
    """Multilayer Perceptron."""

    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor ): You can directly input a ``torch.Tensor``.

        Returns:
            torch.Tensor : ``torch.tensor`` will be returned.
        """
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


def fuse_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
):
    """Fuse object embeddings."""

    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    seq_length = inputs_embeds.shape[1]
    flat_object_embeds = object_embeds.view(-1, object_embeds.shape[-2],
                                            object_embeds.shape[-1])

    valid_object_mask = (
        torch.arange(max_num_objects,
                     device=flat_object_embeds.device)[None, :] <
        num_objects[:, None])

    valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(
        -1, valid_object_embeds.shape[-1])

    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)

    inputs_embeds.masked_scatter_(image_token_mask[:, None],
                                  valid_object_embeds)
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


def build_causal_attention_mask(bsz, seq_len, dtype, device=None):
    """The function originally belonged to CLIPTextTransformer, but it has been
    removed in versions of transformers after 4.25.1."""

    # lazily create causal attention mask,
    # with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype, device=device)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask
