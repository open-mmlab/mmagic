# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmengine import print_log

from mmagic.registry import MODELS


@MODELS.register_module()
class ClipWrapper(nn.Module):
    r"""Clip Models wrapper.

    We provide wrappers for the clip models of ``openai`` and
    ``mlfoundations``, where the user can specify ``clip_type``
    as ``clip`` or ``open_clip``, and then initialize a clip model
    using the same arguments as in the original codebase. The
    following clip models settings are provided in the official
    repo of disco diffusion:
    |            Setting            | Source    | Arguments                                                    | # noqa
    |:-----------------------------:|-----------|--------------------------------------------------------------| # noqa
    | ViTB32                        | clip      | name='ViT-B/32',                  jit=False                  | # noqa
    | ViTB16                        | clip      | name='ViT-B/16',                  jit=False                  | # noqa
    | ViTL14                        | clip      | name='ViT-L/14',                  jit=False                  | # noqa
    | ViTL14_336px                  | clip      | name='ViT-L/14@336px',            jit=False                  | # noqa
    | RN50                          | clip      | name='RN50',                      jit=False                  | # noqa
    | RN50x4                        | clip      | name='RN50x4',                    jit=False                  | # noqa
    | RN50x16                       | clip      | name='RN50x16',                   jit=False                  | # noqa
    | RN50x64                       | clip      | name='RN50x64',                   jit=False                  | # noqa
    | RN101                         | clip      | name='RN101',                     jit=False                  | # noqa
    | ViTB32_laion2b_e16            | open_clip | name='ViT-B-32',                  pretrained='laion2b_e16'   | # noqa
    | ViTB32_laion400m_e31          | open_clip | model_name='ViT-B-32',            pretrained='laion400m_e31' | # noqa
    | ViTB32_laion400m_32           | open_clip | model_name='ViT-B-32',            pretrained='laion400m_e32' | # noqa
    | ViTB32quickgelu_laion400m_e31 | open_clip | model_name='ViT-B-32-quickgelu',  pretrained='laion400m_e31' | # noqa
    | ViTB32quickgelu_laion400m_e32 | open_clip | model_name='ViT-B-32-quickgelu',  pretrained='laion400m_e32' | # noqa
    | ViTB16_laion400m_e31          | open_clip | model_name='ViT-B-16',            pretrained='laion400m_e31' | # noqa
    | ViTB16_laion400m_e32          | open_clip | model_name='ViT-B-16',            pretrained='laion400m_e32' | # noqa
    | RN50_yffcc15m                 | open_clip | model_name='RN50',                pretrained='yfcc15m'       | # noqa
    | RN50_cc12m                    | open_clip | model_name='RN50',                pretrained='cc12m'         | # noqa
    | RN50_quickgelu_yfcc15m        | open_clip | model_name='RN50-quickgelu',      pretrained='yfcc15m'       | # noqa
    | RN50_quickgelu_cc12m          | open_clip | model_name='RN50-quickgelu',      pretrained='cc12m'         | # noqa
    | RN101_yfcc15m                 | open_clip | model_name='RN101',               pretrained='yfcc15m'       | # noqa
    | RN101_quickgelu_yfcc15m       | open_clip | model_name='RN101-quickgelu',     pretrained='yfcc15m'       | # noqa

    An example of a ``clip_modes_cfg`` is as follows:

    Examples:

    >>> # Use OpenAI's CLIP
    >>> config = dict(
    >>>     type='ClipWrapper',
    >>>     clip_type='clip',
    >>>     name='ViT-B/32',
    >>>     jit=False)

    >>> # Use OpenCLIP
    >>> config = dict(
    >>>     type='ClipWrapper',
    >>>     clip_type='open_clip',
    >>>     model_name='RN50',
    >>>     pretrained='yfcc15m')

    >>> # Use CLIP from Hugging Face Transformers
    >>> config = dict(
    >>>     type='ClipWrapper',
    >>>     clip_type='huggingface',
    >>>     pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
    >>>     subfolder='text_encoder')

    Args:
        clip_type (List[Dict]): The original source of the clip model. Whether be
            ``clip``, ``open_clip`` or ``hugging_face``.

        *args, **kwargs: Arguments to initialize corresponding clip model.
    """

    def __init__(self, clip_type, *args, **kwargs):

        super().__init__()
        self.clip_type = clip_type
        assert clip_type in ['clip', 'open_clip', 'huggingface']

        error_msg = ('{} need to be installed! Run `pip install -r '
                     'requirements/optional.txt` and try again')
        if clip_type == 'clip':
            try:
                import clip
            except ImportError:
                raise ImportError(error_msg.format('\'clip\''))
            print_log(f'Creating {kwargs["name"]} by OpenAI', 'current')
            self.model, _ = clip.load(*args, **kwargs)
        elif clip_type == 'open_clip':
            try:
                import open_clip
            except ImportError:
                raise ImportError(error_msg.format('\'open_clip_torch\''))
            print_log(f'Creating {kwargs["model_name"]} by '
                      'mlfoundations', 'current')
            self.model = open_clip.create_model(*args, **kwargs)

        elif clip_type == 'huggingface':
            try:
                import transformers
            except ImportError:
                raise ImportError(error_msg.format('\'transforms\''))
            # NOTE: use CLIPTextModel to adopt stable diffusion pipeline
            model_cls = transformers.CLIPTextModel
            self.model = model_cls.from_pretrained(*args, **kwargs)
            self.config = self.model.config
            print_log(
                f'Creating {self.model.name_or_path} '
                'by \'HuggingFace\'', 'current')

        self.model.eval().requires_grad_(False)

    def get_embedding_layer(self):
        """Function to get embedding layer of the clip model.

        Only support for CLIPTextModel currently.
        """

        if self.clip_type != 'huggingface':
            print_log(
                'Do not support \'get_embedding_layer\' for clip_type: '
                f'\'{self.clip_type}\' currently.', 'current')
            return None
        if self.model.__class__.__name__ != 'CLIPTextModel':
            print_log(
                'Only support \'get_embedding_layer\' for '
                'CLIPTextModel.', 'current')
            return None

        return self.model.text_model.embeddings.token_embedding

    def add_embedding(self, embeddings: Union[dict, List[dict]]):
        assert self.clip_type == 'huggingface', (
            'Only support add embedding for HuggingFace transformers.')
        assert self.model.__class__.__name__ == 'CLIPTextModel', (
            'Only support add embedding for \'CLIPTextModel\' (CLIP).')

        embedding_layer = self.get_embedding_layer()
        if not isinstance(embedding_layer, EmbeddingLayerWithFixes):
            self.model.embeddings = EmbeddingLayerWithFixes(embedding_layer)

        self.model.embeddings.add_embedding(embeddings)

    def set_only_embedding_trainable(self):
        func_name = '\'set_only_embedding_trainable\''
        assert self.clip_type == 'huggingface', (
            f'Only support {func_name} for HuggingFace transformers.')
        assert self.model.__class__.__name__ == 'CLIPTextModel', (
            f'Only support {func_name} for \'CLIPTextModel\' (CLIP).')
        self.model.requires_grad_(False)
        embedding_layer = self.get_embedding_layer()
        if isinstance(embedding_layer, EmbeddingLayerWithFixes):
            embedding_layer.trainable_embeddings.requires_grad_(True)
            print_log('Set only embedding trainable.', 'current')
        else:
            print_log(
                'Do not found EmbeddingLayerWithFixes. '
                f'{func_name} do nothing.', 'current')

    def set_embedding_layer(self):
        assert self.clip_type == 'huggingface', (
            'Only support add embedding for HuggingFace transformers.')
        assert self.model.__class__.__name__ == 'CLIPTextModel', (
            'Only support add embedding for \'CLIPTextModel\' (CLIP).')
        embedding_layer = self.get_embedding_layer()
        if not isinstance(embedding_layer, EmbeddingLayerWithFixes):
            self.model.text_model.embeddings.token_embedding = \
                EmbeddingLayerWithFixes(embedding_layer)
        print_log('Set embedding layer to EmbeddingLayerWithFixes', 'current')

    def unset_embedding_layer(self):
        wrapped_embedding_layer = self.model.embeddings
        if isinstance(wrapped_embedding_layer, EmbeddingLayerWithFixes):
            self.model.text_model.embeddings.token_embedding = \
                wrapped_embedding_layer.wrapped
        print_log('Unset embedding layer.', 'current')

    def forward(self, *args, **kwargs):
        """Forward function."""
        return self.model(*args, **kwargs)


class EmbeddingLayerWithFixes(nn.Module):
    """The revised embedding layer to support external embeddings. This design
    of this class is inspired by https://github.com/AUTOMATIC1111/stable-
    diffusion-webui/blob/22bcc7be428c94e9408f589966c2040187245d81/modules/sd_hi
    jack.py#L224  # noqa.

    Args:
        wrapped (nn.Emebdding): The embedding layer to be wrapped.
        external_embeddings (Union[dict, List[dict]], optional): The external
            embeddings added to this layer. Defaults to None.
    """

    def __init__(self,
                 wrapped: nn.Embedding,
                 external_embeddings: Optional[Union[dict,
                                                     List[dict]]] = None):
        super().__init__()
        self.wrapped = wrapped
        self.num_embeddings = wrapped.weight.shape[0]

        self.external_embeddings = []
        if external_embeddings:
            self.add_embeddings(external_embeddings)

        self.trainable_embeddings = nn.ParameterDict()

    @property
    def weight(self):
        """Get the weight of wrapped embedding layer."""
        return self.wrapped.weight

    def check_duplicate_names(self, embeddings: List[dict]):
        """Check whether duplicate names exist in list of 'external
        embeddings'.

        Args:
            embeddings (List[dict]): A list of embedding to be check.
        """
        names = [emb['name'] for emb in embeddings]
        assert len(names) == len(set(names)), (
            'Found duplicated names in \'external_embeddings\'. Name list: '
            f'\'{names}\'')

    def check_ids_overlap(self, embeddings):
        """Check whether overlap exist in token ids of 'external_embeddings'.

        Args:
            embeddings (List[dict]): A list of embedding to be check.
        """
        ids_range = [[emb['start'], emb['end'], emb['name']]
                     for emb in embeddings]
        ids_range.sort()  # sort by 'start'
        # check if 'end' has overlapping
        for idx in range(len(ids_range) - 1):
            name1, name2 = ids_range[idx][-1], ids_range[idx + 1][-1]
            assert ids_range[idx][1] <= ids_range[idx + 1][0], (
                f'Found ids overlapping between embeddings \'{name1}\' '
                f'and \'{name2}\'.')

    def add_embeddings(self, embeddings: Optional[Union[dict, List[dict]]]):
        """Add external embeddings to this layer.

        Use case:

        >>> 1. Add token to tokenizer and get the token id.
        >>> tokenizer = TokenizerWrapper('openai/clip-vit-base-patch32')
        >>> # 'how much' in kiswahili
        >>> tokenizer.add_placeholder_tokens('ngapi', num_vec_per_token=4)
        >>>
        >>> 2. Add external embeddings to the model.
        >>> new_embedding = {
        >>>     'name': 'ngapi',  # 'how much' in kiswahili
        >>>     'embedding': torch.ones(1, 15) * 4,
        >>>     'start': tokenizer.get_token_info('kwaheri')['start'],
        >>>     'end': tokenizer.get_token_info('kwaheri')['end'],
        >>>     'trainable': False  # if True, will registry as a parameter
        >>> }
        >>> embedding_layer = nn.Embedding(10, 15)
        >>> embedding_layer_wrapper = EmbeddingLayerWithFixes(embedding_layer)
        >>> embedding_layer_wrapper.add_embeddings(new_embedding)
        >>>
        >>> 3. Forward tokenizer and embedding layer!
        >>> input_text = ['hello, ngapi!', 'hello my friend, ngapi?']
        >>> input_ids = tokenizer(
        >>>     input_text, padding='max_length', truncation=True,
        >>>     return_tensors='pt')['input_ids']
        >>> out_feat = embedding_layer_wrapper(input_ids)
        >>>
        >>> 4. Let's validate the result!
        >>> assert (out_feat[0, 3: 7] == 2.3).all()
        >>> assert (out_feat[2, 5: 9] == 2.3).all()

        Args:
            embeddings (Union[dict, list[dict]]): The external embeddings to
                be added. Each dict must contain the following 4 fields: 'name'
                (the name of this embedding), 'embedding' (the embedding
                tensor), 'start' (the start token id of this embedding), 'end'
                (the end token id of this embedding). For example:
                `{name: NAME, start: START, end: END, embedding: torch.Tensor}`
        """
        if isinstance(embeddings, dict):
            embeddings = [embeddings]

        self.external_embeddings += embeddings
        self.check_duplicate_names(self.external_embeddings)
        self.check_ids_overlap(self.external_embeddings)

        # set for trainable
        added_trainable_emb_info = []
        for embedding in embeddings:
            trainable = embedding.get('trainable', False)
            if trainable:
                name = embedding['name']
                embedding['embedding'] = torch.nn.Parameter(
                    embedding['embedding'])
                self.trainable_embeddings[name] = embedding['embedding']
                added_trainable_emb_info.append(name)

        added_emb_info = [emb['name'] for emb in embeddings]
        added_emb_info = ', '.join(added_emb_info)
        print_log(f'Successfully add external embeddings: {added_emb_info}.',
                  'current')

        if added_trainable_emb_info:
            added_trainable_emb_info = ', '.join(added_trainable_emb_info)
            print_log(
                'Successfully add trainable external embeddings: '
                f'{added_trainable_emb_info}', 'current')

    def replace_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace external input ids to 0.

        Args:
            input_ids (torch.Tensor): The input ids to be replaced.

        Returns:
            torch.Tensor: The replaced input ids.
        """
        input_ids_fwd = input_ids.clone()
        input_ids_fwd[input_ids_fwd >= self.num_embeddings] = 0
        return input_ids_fwd

    def replace_embeddings(self, input_ids: torch.Tensor,
                           embedding: torch.Tensor,
                           external_embedding: dict) -> torch.Tensor:
        """Replace external embedding to the embedding layer. Noted that, in
        this function we use `torch.cat` to avoid inplace modification.

        Args:
            input_ids (torch.Tensor): The original token ids. Shape like
                [LENGTH, ].
            embedding (torch.Tensor): The embedding of token ids after
                `replace_input_ids` function.
            external_embedding (dict): The external embedding to be replaced.

        Returns:
            torch.Tensor: The replaced embedding.
        """
        new_embedding = []

        name = external_embedding['name']
        start = external_embedding['start']
        end = external_embedding['end']
        target_ids_to_replace = [i for i in range(start, end)]
        ext_emb = external_embedding['embedding']

        # do not need to replace
        if not (input_ids == start).any():
            return embedding

        # start replace
        s_idx, e_idx = 0, 0
        while e_idx < len(input_ids):
            if input_ids[e_idx] == start:
                if e_idx != 0:
                    # add embedding do not need to replace
                    new_embedding.append(embedding[s_idx:e_idx])

                # check if the next embedding need to replace is valid
                actually_ids_to_replace = [
                    int(i) for i in input_ids[e_idx:e_idx + end - start]
                ]
                assert actually_ids_to_replace == target_ids_to_replace, (
                    f'Invalid \'input_ids\' in position: {s_idx} to {e_idx}. '
                    f'Expect \'{target_ids_to_replace}\' for embedding '
                    f'\'{name}\' but found \'{actually_ids_to_replace}\'.')

                new_embedding.append(ext_emb)

                s_idx = e_idx + end - start
                e_idx = s_idx + 1
            else:
                e_idx += 1

        if e_idx == len(input_ids):
            new_embedding.append(embedding[s_idx:e_idx])

        return torch.cat(new_embedding, dim=0)

    def forward(self,
                input_ids: torch.Tensor,
                external_embeddings: Optional[List[dict]] = None):
        """The forward function.

        Args:
            input_ids (torch.Tensor): The token ids shape like [bz, LENGTH] or
                [LENGTH, ].
            external_embeddings (Optional[List[dict]]): The external
                embeddings. If not passed, only `self.external_embeddings`
                will be used.  Defaults to None.

        input_ids: shape like [bz, LENGTH] or [LENGTH].
        """
        assert input_ids.ndim in [1, 2]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        if external_embeddings is None and not self.external_embeddings:
            return self.wrapped(input_ids)

        input_ids_fwd = self.replace_input_ids(input_ids)
        inputs_embeds = self.wrapped(input_ids_fwd)

        vecs = []

        if external_embeddings is None:
            external_embeddings = []
        elif isinstance(external_embeddings, dict):
            external_embeddings = [external_embeddings]
        embeddings = self.external_embeddings + external_embeddings

        for input_id, embedding in zip(input_ids, inputs_embeds):
            new_embedding = embedding
            for external_embedding in embeddings:
                new_embedding = self.replace_embeddings(
                    input_id, new_embedding, external_embedding)
            vecs.append(new_embedding)

        return torch.stack(vecs)
