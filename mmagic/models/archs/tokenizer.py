# Copyright (c) OpenMMLab. All rights reserved.
"""This a wrapper for tokenizer."""
import copy
import os
import random
from logging import WARNING
from typing import Any, List, Optional, Union

from mmengine import print_log

from mmagic.utils import try_import


class TokenizerWrapper:
    """Tokenizer wrapper for CLIPTokenizer. Only support CLIPTokenizer
    currently. This wrapper is modified from https://github.com/huggingface/dif
    fusers/blob/e51f19aee82c8dd874b715a09dbc521d88835d68/src/diffusers/loaders.
    py#L358  # noqa.

    Args:
        from_pretrained (Union[str, os.PathLike], optional): The *model id*
            of a pretrained model or a path to a *directory* containing
            model weights and config. Defaults to None.
        from_config (Union[str, os.PathLike], optional): The *model id*
            of a pretrained model or a path to a *directory* containing
            model weights and config. Defaults to None.

        *args, **kwargs: If `from_pretrained` is passed, *args and **kwargs
            will be passed to `from_pretrained` function. Otherwise, *args
            and **kwargs will be used to initialize the model by
            `self._module_cls(*args, **kwargs)`.
    """

    def __init__(self,
                 from_pretrained: Optional[Union[str, os.PathLike]] = None,
                 from_config: Optional[Union[str, os.PathLike]] = None,
                 *args,
                 **kwargs):
        transformers = try_import('transformers')
        module_cls = transformers.CLIPTokenizer

        assert not (from_pretrained and from_config), (
            '\'from_pretrained\' and \'from_config\' should not be passed '
            'at the same time.')

        if from_config:
            print_log(
                'Tokenizers from Huggingface transformers do not support '
                '\'from_config\'. Will call \'from_pretrained\' instead '
                'with the same argument.', 'current', WARNING)
            from_pretrained = from_config

        if from_pretrained:
            self.wrapped = module_cls.from_pretrained(from_pretrained, *args,
                                                      **kwargs)
        else:
            self.wrapper = module_cls(*args, **kwargs)

        self._from_pretrained = from_pretrained
        self.token_map = {}

    def __getattr__(self, name: str) -> Any:
        if name == 'wrapped':
            return super().__getattr__('wrapped')

        try:
            return getattr(self.wrapped, name)
        except AttributeError:
            try:
                return super().__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    '\'name\' cannot be found in both '
                    f'\'{self.__class__.__name__}\' and '
                    f'\'{self.__class__.__name__}.tokenizer\'.')

    def try_adding_tokens(self, tokens: Union[str, List[str]], *args,
                          **kwargs):
        """Attempt to add tokens to the tokenizer.

        Args:
            tokens (Union[str, List[str]]): The tokens to be added.
        """
        num_added_tokens = self.wrapped.add_tokens(tokens, *args, **kwargs)
        assert num_added_tokens != 0, (
            f'The tokenizer already contains the token {tokens}. Please pass '
            'a different `placeholder_token` that is not already in the '
            'tokenizer.')

    def get_token_info(self, token: str) -> dict:
        """Get the information of a token, including its start and end index in
        the current tokenizer.

        Args:
            token (str): The token to be queried.

        Returns:
            dict: The information of the token, including its start and end
                index in current tokenizer.
        """
        token_ids = self.__call__(token).input_ids
        start, end = token_ids[1], token_ids[-2] + 1
        return {'name': token, 'start': start, 'end': end}

    def add_placeholder_token(self,
                              placeholder_token: str,
                              *args,
                              num_vec_per_token: int = 1,
                              **kwargs):
        """Add placeholder tokens to the tokenizer.

        Args:
            placeholder_token (str): The placeholder token to be added.
            num_vec_per_token (int, optional): The number of vectors of
                the added placeholder token.
            *args, **kwargs: The arguments for `self.wrapped.add_tokens`.
        """
        output = []
        if num_vec_per_token == 1:
            self.try_adding_tokens(placeholder_token, *args, **kwargs)
            output.append(placeholder_token)
        else:
            output = []
            for i in range(num_vec_per_token):
                ith_token = placeholder_token + f'_{i}'
                self.try_adding_tokens(ith_token, *args, **kwargs)
                output.append(ith_token)

        for token in self.token_map:
            if token in placeholder_token:
                raise ValueError(
                    f'The tokenizer already has placeholder token {token} '
                    f'that can get confused with {placeholder_token} '
                    'keep placeholder tokens independent')
        self.token_map[placeholder_token] = output

    def replace_placeholder_tokens_in_text(self,
                                           text: Union[str, List[str]],
                                           vector_shuffle: bool = False,
                                           prop_tokens_to_load: float = 1.0
                                           ) -> Union[str, List[str]]:
        """Replace the keywords in text with placeholder tokens. This function
        will be called in `self.__call__` and `self.encode`.

        Args:
            text (Union[str, List[str]]): The text to be processed.
            vector_shuffle (bool, optional): Whether to shuffle the vectors.
                Defaults to False.
            prop_tokens_to_load (float, optional): The proportion of tokens to
                be loaded. If 1.0, all tokens will be loaded. Defaults to 1.0.

        Returns:
            Union[str, List[str]]: The processed text.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(
                    self.replace_placeholder_tokens_in_text(
                        text[i], vector_shuffle=vector_shuffle))
            return output

        for placeholder_token in self.token_map:
            if placeholder_token in text:
                tokens = self.token_map[placeholder_token]
                tokens = tokens[:1 + int(len(tokens) * prop_tokens_to_load)]
                if vector_shuffle:
                    tokens = copy.copy(tokens)
                    random.shuffle(tokens)
                text = text.replace(placeholder_token, ' '.join(tokens))
        return text

    def replace_text_with_placeholder_tokens(self, text: Union[str, List[str]]
                                             ) -> Union[str, List[str]]:
        """Replace the placeholder tokens in text with the original keywords.
        This function will be called in `self.decode`.

        Args:
            text (Union[str, List[str]]): The text to be processed.

        Returns:
            Union[str, List[str]]: The processed text.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(
                    self.replace_text_with_placeholder_tokens(text[i]))
            return output

        for placeholder_token, tokens in self.token_map.items():
            merged_tokens = ' '.join(tokens)
            if merged_tokens in text:
                text = text.replace(merged_tokens, placeholder_token)
        return text

    def __call__(self,
                 text: Union[str, List[str]],
                 *args,
                 vector_shuffle: bool = False,
                 prop_tokens_to_load: float = 1.0,
                 **kwargs):
        """The call function of the wrapper.

        Args:
            text (Union[str, List[str]]): The text to be tokenized.
            vector_shuffle (bool, optional): Whether to shuffle the vectors.
                Defaults to False.
            prop_tokens_to_load (float, optional): The proportion of tokens to
                be loaded. If 1.0, all tokens will be loaded. Defaults to 1.0
            *args, **kwargs: The arguments for `self.wrapped.__call__`.
        """
        replaced_text = self.replace_placeholder_tokens_in_text(
            text,
            vector_shuffle=vector_shuffle,
            prop_tokens_to_load=prop_tokens_to_load)

        return self.wrapped.__call__(replaced_text, *args, **kwargs)

    def encode(self, text: Union[str, List[str]], *args, **kwargs):
        """Encode the passed text to token index.

        Args:
            text (Union[str, List[str]]): The text to be encode.
            *args, **kwargs: The arguments for `self.wrapped.__call__`.
        """
        replaced_text = self.replace_placeholder_tokens_in_text(text)
        return self.wrapped(replaced_text, *args, **kwargs)

    def decode(self,
               token_ids,
               return_raw: bool = False,
               *args,
               **kwargs) -> Union[str, List[str]]:
        """Decode the token index to text.

        Args:
            token_ids: The token index to be decoded.
            return_raw: Whether keep the placeholder token in the text.
                Defaults to False.
            *args, **kwargs: The arguments for `self.wrapped.decode`.

        Returns:
            Union[str, List[str]]: The decoded text.
        """
        text = self.wrapped.decode(token_ids, *args, **kwargs)
        if return_raw:
            return text
        replaced_text = self.replace_text_with_placeholder_tokens(text)
        return replaced_text

    def __repr__(self):
        """The representation of the wrapper."""
        s = super().__repr__()
        prefix = f'Wrapped Module Class: {self._module_cls}\n'
        prefix += f'Wrapped Module Name: {self._module_name}\n'
        if self._from_pretrained:
            prefix += f'From Pretrained: {self._from_pretrained}\n'
        s = prefix + s
        return s
