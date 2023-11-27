# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Any, List, Optional, Union

import torch.nn as nn
from mmengine import print_log
from torch import Tensor


class LoRALinear(nn.Module):
    """Linear layer for LoRA.

    Args:
        in_feat (int): Number of input features.
        out_feat (int): Number of output features.
        rank (int): The rank of LoRA.
    """

    def __init__(self, in_feat: int, out_feat: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        assert rank < min(in_feat, out_feat)

        self.down = nn.Linear(in_feat, rank, bias=False)
        self.up = nn.Linear(rank, out_feat, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        ori_type = x.dtype
        dtype = self.down.weight.dtype

        out = self.down(x.to(dtype))
        out = self.up(out)
        return out.to(ori_type)


class LoRAWrapper(nn.Module):
    """Wrapper for LoRA layer.

    Args:
        module (nn.Module): The module to be wrapped.
        in_feat (int): Number of input features.
        out_feat (int): Number of output features.
        rank (int): The rank of LoRA.
        scale (float): The scale of LoRA feature.
        names (Union[str, List[str]], optional): The name of LoRA layers. If
            you want to add multi LoRA for one module, names for each LoRA
            mapping must be defined.
    """

    def __init__(self,
                 module: nn.Module,
                 in_feat: int,
                 out_feat: int,
                 rank: int,
                 scale: float = 1,
                 names: Optional[Union[str, List[str]]] = None):
        super().__init__()

        # NOTE: LoRA for linear layer, LoCON will coming soon~
        assert isinstance(
            module,
            nn.Linear), ('Only Support LoRA for linear layer currently. '
                         'LoCON will coming soon~')
        self.wrapped = module

        if names is not None:
            # set a list of LoRAs
            if not isinstance(names, list):
                names = [names]
            if isinstance(rank, list):
                assert len(rank) == len(names)
            else:
                rank = [rank] * len(names)
            if isinstance(scale, list):
                assert len(scale) == len(names)
            else:
                scale = [scale] * len(names)

            self.names = names
            self.lora_mapping = dict()
            self.scale = dict()
            self.enable = dict()
            self.rank = dict()
            for n, r, s in zip(names, rank, scale):
                self.lora_mapping[n] = LoRALinear(in_feat, out_feat, r)
                self.scale_dict[n] = s
                self.enable[n] = True
                self.rank[n] = r
            self.lora_mapping = nn.ModuleDict(self.lora_mapping)

        else:
            # set single LoRA
            self.names = None
            self.lora_mapping = LoRALinear(in_feat, out_feat, rank)
            self.scale = scale
            self.enable = True
            self.rank = rank

        self.in_feat, self.out_feat = in_feat, out_feat

    def add_lora(self,
                 name: str,
                 rank: int,
                 scale: float = 1,
                 state_dict: Optional[dict] = None):
        """Add LoRA mapping.

        Args:
            name (str): The name of added LoRA.
            rank (int): The rank of added LoRA.
            scale (float, optional): The scale of added LoRA. Defaults to 1.
            state_dict (dict, optional): The state dict of added LoRA.
                Defaults to None.
        """
        mapping_to_add = LoRALinear(self.in_feat, self.out_feat, rank)
        if state_dict is not None:
            mapping_to_add.load_state_dict(mapping_to_add)
        # move to device and type
        mapping_to_add.to(self.lora_mapping.weight.dtype)

        if isinstance(self.names, list):
            self.names.append(name)
            self.lora_mapping[name] = mapping_to_add
            self.scale[name] = scale
            self.enable[name] = True
            self.rank[name] = rank
        else:
            self.names = ['orig', name]
            self.lora_mapping = nn.ModuleDict({
                'orig': self.lora_mapping,
                name: mapping_to_add
            })
            self.scale = {'orig': self.scale, name: scale}
            self.enable = {'orig': self.enable, name: True}
            self.rank = {'orig': self.rank, name: rank}
            print_log(
                'The original LoRA mapping do not have name, '
                'save as \'orig\'.', 'current')
        print_log(f'Add LoRA \'{name}\' with rank {rank} and scale {scale}.',
                  'current')

    def _set_value(self,
                   attr_name: str,
                   value: Any,
                   name: Optional[str] = None):
        """Set value of attribute.

        Args:
            attr_name (str): The name of attribute to be set value.
            value (Any): The value to be set.
            name (str, optional): The name of field in `attr_name`. If
                passed, will set value to `attr_name[name]`. Defaults to None.
        """
        attr = getattr(self, attr_name)

        if isinstance(attr, dict):
            if name is None:
                attr = {k: value for k in self.names}
                print_log(f'Set all value in \'{attr_name}\' as \'{value}\'.',
                          'current')
            else:
                attr[name] = value
                print_log(f'Set \'{attr_name}[{name}]\' as \'{value}\'.',
                          'current')
        else:
            attr = value
            print_log(f'Set \'{attr_name}\' as \'{value}\'.', 'current')

        setattr(self, attr_name, attr)

    def set_scale(self, scale: float, name: Optional[str] = None):
        """Set LoRA scale.

        Args:
            scale (float): The scale to be set.
            name (str, optional): The name of LoRA to be set. Defaults to None.
        """
        self._set_value('scale', scale, name)

    def set_enable(self, name: Optional[str] = None):
        """Enable LoRA for the current layer.

        Args:
            name (str, optional): The name of LoRA to be set. Defaults to None.
        """
        self._set_value('enable', True, name)

    def set_disable(self, name: Optional[str] = None):
        """Disable LoRA for the current layer.

        Args:
            name (str, optional): The name of LoRA to be set. Defaults to None.
        """
        self._set_value('enable', False, name)

    def forward_lora_mapping(self, x: Tensor) -> Tensor:
        """Forward LoRA mapping.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        mapping_out = 0
        if isinstance(self.lora_mapping, dict):
            for name in self.names:
                scale = self.scale[name]
                mapping_layer = self.lora_mapping[name]
                enable = self.enable[name]

                if enable:
                    mapping_out = scale * mapping_layer(x)
        else:
            if self.enable:
                mapping_out = self.scale * self.lora_mapping(x)
        return mapping_out

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward and add LoRA mapping.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        mapping_out = self.forward_lora_mapping(x)
        return mapping_out + self.wrapped(x)

    @classmethod
    def wrap_lora(cls, module, rank=4, scale=1, names=None, state_dict=None):
        """Wrap LoRA.

        Use case:
        >>> linear = nn.Linear(2, 4)
        >>> lora_linear = LoRAWrapper.wrap_lora(linear, 4, 1)

        Args:
            module (nn.Module): The module to add LoRA.
            rank (int): The rank for LoRA.
            scale (float):

        Returns:
            LoRAWrapper:
        """
        assert isinstance(module,
                          nn.Linear), 'Only support LoRA for Linear Layer'
        in_feat = module.weight.shape[1]
        out_feat = module.weight.shape[0]
        lora = LoRAWrapper(module, in_feat, out_feat, rank, scale, names)

        return lora


def replace_module(parent_module: nn.Module, child_name: str,
                   new_module: nn.Module):
    """Replace module in parent module."""
    setattr(parent_module, child_name, new_module)


def get_submodule(module: nn.Module, key: str):
    """Get submodule by key."""
    target_name = key.split('.')[-1]
    parent = module.get_submodule('.'.join(key.split('.')[:-1]))
    target = module.get_submodule(key)
    return parent, target, target_name


def set_lora(module: nn.Module,
             config: dict,
             verbose: bool = True) -> nn.Module:
    """Set LoRA for module.

    Use case:
    >>> 1. set all lora with same parameters
    >>> lora_config = dict(
    >>>     rank=4,
    >>>     scale=1,
    >>>     target_modules=['to_q', 'to_k', 'to_v'])

    >>> 2. set lora with different parameters
    >>> lora_config = dict(
    >>>     rank=4,
    >>>     scale=1,
    >>>     target_modules=[
    >>>         # set `to_q` the default parameters
    >>>         'to_q',
    >>>         # set `to_k` the defined parameters
    >>>         dict(target_module='to_k', rank=8, scale=1),
    >>>         # set `to_v` the defined `rank` and default `scale`
    >>>         dict(target_module='to_v', rank=16)
    >>>     ])

    Args:
        module (nn.Module): The module to set LoRA.
        config (dict): The config dict.
        verbose (bool): Whether to print log. Defaults to True.
    """
    default_rank = config.get('rank', 4)
    default_scale = config.get('scale', 1)
    target_modules = config['target_modules']
    if not isinstance(target_modules, list):
        target_modules = [target_modules]

    keys = [k for k, _ in module.named_modules()]

    for k in keys:
        for target_module in target_modules:
            if isinstance(target_module, str):
                module_name = target_module
                rank = default_rank
                scale = default_scale
                # pretrained_path = None

            elif isinstance(target_module, dict):
                module_name = target_module['target_module']
                rank = target_module.get('rank', default_rank)
                scale = target_module.get('scale', default_scale)
                # pretrained_path = target_module.get('pretrained_path', None)

            else:
                raise TypeError('Only support dict or string type for '
                                'target_modules')
            # match keys
            if re.fullmatch(module_name, k):
                if verbose:
                    print_log(
                        f'Set LoRA for \'{k}\' with '
                        f'regularization expression match \'{module_name}\'.',
                        'current')
            elif k.endswith(module_name):
                if verbose:
                    print_log(
                        f'Set LoRA for \'{k}\' with '
                        f'suffix match \'{module_name}\'.', 'current')
            else:
                continue

            parent, target, target_name = get_submodule(module, k)
            new_module = LoRAWrapper.wrap_lora(target, rank=rank, scale=scale)
            replace_module(parent, target_name, new_module)
    return module


def set_only_lora_trainable(module: nn.Module) -> nn.Module:
    """Set only LoRA modules trainable."""
    for n, m in module.named_children():
        if isinstance(m, LoRAWrapper):
            m.lora_mapping.requires_grad_(True)
        elif isinstance(m, nn.Module):
            m.requires_grad_(False)
            set_only_lora_trainable(m)

    return module


def set_lora_enable(module: nn.Module) -> nn.Module:
    """Enable LoRA modules."""
    for n, m in module.named_children():
        if isinstance(m, LoRAWrapper):
            m.set_enable()
        elif isinstance(m, nn.Module):
            set_lora_enable(m)
    return module


def set_lora_disable(module: nn.Module) -> nn.Module:
    """Disable LoRA modules."""
    for n, m in module.named_children():
        if isinstance(m, LoRAWrapper):
            m.set_disable()
        elif isinstance(m, nn.Module):
            set_lora_disable(m)
    return module
