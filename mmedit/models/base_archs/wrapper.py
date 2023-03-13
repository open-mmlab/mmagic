# Copyright (c) OpenMMLab. All rights reserved.
import os
from copy import deepcopy
from typing import Any, List, Optional, Union
from warnings import warn

from mmengine.model import BaseModule


class DiffuserWrapper(BaseModule):
    """Wrapper for models from HuggingFace Diffusers. This wrapper will be set
    a attribute called `_module_cls` by wrapping function and will be used to
    initialize the model structure.

    Example:
    >>> 1. Load pretrained model from HuggingFace Space.
    >>> config = dict(
    >>>     type='ControlNetModel',  # has been registered in `MODELS`
    >>>     repo_id='lllyasviel/sd-controlnet-canny')
    >>> controlnet = MODELS.build(config)

    >>> 2. Initialize model structure but do not load pretrained weight
    >>> config = dict(
    >>>     type='ControlNetModel',  # has been registered in `MODELS`
    >>>     repo_id='lllyasviel/sd-controlnet-canny',
    >>>     no_loading=True)
    >>> controlnet = MODELS.build(config)

    >>> 3. Loading pretrained model with specific settings (e.g., fp16).
    >>> config = dict(
    >>>     type='ControlNetModel',  # has been registered in `MODELS`
    >>>     repo_id='lllyasviel/sd-controlnet-canny',
    >>>     init_cfg=dict(type='Pretrained', torch_dtype=torch.float16))
    >>> controlnet = MODELS.build(config)

    >>> 4. Initialize model with own defined arguments
    >>> config = dict(
    >>>     type='ControlNetModel',
    >>>     in_channels=3,
    >>>     down_block_types=['DownBlock2D'],
    >>>     block_out_channels=(20, ),
    >>>     conditioning_embedding_out_channels=(16, ))
    >>> controlnet = MODELS.build(config)

    Args:
        repo_id (Union[str, os.PathLike], optional): The *model id* or path to
            pretrained model. Please refers to `diffusers.model.modeling_utils.ModelMixin.from_pretrained`.  # noqa
            If passed, the structure of the model will be initialized as model
            structure of `repo_id`. Defaults to None.
        no_loading (bool): Whether loading the pretrained weights of `repo_id`.
            If True, weight of `repo_id` will not be loaded. Defaults to False.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Noted that, in `DiffuserWrapper`, if you want to load pretrained
            weight of `repo_id`, please use `no_loading=False`. If you want to
            modify the `from_pretrained` behavior, you should set `type` as
            `Pretrained` and set corresponding arguments in `init_cfg`.
            e.g. (`dict(type='Pretrained', cache_dir='~/.cache/OpenMMLab/')`)
            Defaults to None.

        *args, **kwargs: Arguments for `module_cls`.
    """

    def __init__(self,
                 repo_id: Optional[Union[str, os.PathLike]] = None,
                 no_loading: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None,
                 *args,
                 **kwargs):
        super().__init__(init_cfg)

        self._repo_id = repo_id
        self._no_loading = no_loading

        module_cls = self._module_cls

        if repo_id is not None:

            if no_loading:
                _config = module_cls.load_config(repo_id)
                self.model = module_cls(**_config)
            else:
                if init_cfg and init_cfg['type'] == 'Pretrained':
                    from_pretrained_args = deepcopy(init_cfg)
                    from_pretrained_args.pop('type')
                else:
                    from_pretrained_args = dict()

                self.model = module_cls.from_pretrained(
                    repo_id, **from_pretrained_args)
                self._is_init = True
        else:
            self.model = module_cls(*args, **kwargs)

        self.config = self.model.config
        self.init_weights()

    def init_weights(self):
        """Initialize the weights.

        If type is 'Pretrained' but the model has be loaded from `repo_id`, a
        warning will be raised.
        """

        if self.init_cfg and self.init_cfg['type'] == 'Pretrained':
            if self._repo_id and not self._no_loading:
                warn('Has been loaded from pretrained model. '
                     'You behavior is very dangerous.')
        super().init_weights()

    def __getattr__(self, name: str) -> Any:
        """This function provide a way to access the attributes of the wrapped
        model.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The got attribute.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self.model, name)
            except AttributeError:
                raise AttributeError('\'name\' cannot be found in both '
                                     f'\'{self.__class__.__name__}\' and '
                                     f'\'{self.__class__.__name__}.model\'.')

    def __repr__(self):
        """The representation of the wrapper."""
        s = super().__repr__()
        prefix = f'Wrapped Module Class: {self._module_cls}\n'
        prefix += f'Wrapped Module Name: {self._module_name}\n'
        if self._repo_id:
            prefix += f'Repo ID: {self._repo_id}\n'
        s = prefix + s
        return s
