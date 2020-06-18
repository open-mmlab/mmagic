from collections import OrderedDict

import torch
import torch.nn as nn
from mmcv.parallel import MODULE_WRAPPERS, MMDistributedDataParallel
from mmcv.parallel.scatter_gather import scatter_kwargs
from torch.nn.parallel import DataParallel, DistributedDataParallel


@MODULE_WRAPPERS.register_module()
class DistributedDataParallelWrapper(nn.Module):
    """A DistributedDataParallel wrapper for models in MMediting.

    In MMedting, there is a need to wrap different modules in the models
    with separate DistributedDataParallel. Otherwise, it will cause
    errors for GAN training.
    More specific, the GAN model, usually has two sub-modules:
    generator and discriminator. If we wrap both of them in one
    standard DistributedDataParallel, it will cause errors during training,
    because when we update the parameters of the generator (or discriminator),
    the parameters of the discriminator (or generator) is not updated, which is
    not allowed for DistributedDataParallel.
    So we design this wrapper to separately wrap DistributedDataParallel
    for generator and discriminator.

    In this wrapper, we perform two operations:
    1. Wrap the modules in the models with separate MMDistributedDataParallel.
        Note that only modules with parameters will be wrapped.
    2. Do scatter operation for 'forward', 'train_step' and 'val_step'.

    Note that the arguments of this wrapper is the same as those in
    `torch.nn.parallel.distributed.DistributedDataParallel`.

    Args:
        module (nn.Module): Module that needs to be wrapped.
        device_ids (list[int | `torch.device`]): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
        dim (int, optional): Same as that in the official scatter function in
            pytorch. Defaults to 0.
        broadcast_buffers (bool): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
            Defaults to False.
        find_unused_parameters (bool, optional): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
            Traverse the autograd graph of all tensors contained in returned
            value of the wrapped module’s forward function. Defaults to False.
        kwargs (dict): Other arguments used in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
    """

    def __init__(self,
                 module,
                 device_ids,
                 dim=0,
                 broadcast_buffers=False,
                 find_unused_parameters=False,
                 **kwargs):
        super(DistributedDataParallelWrapper, self).__init__()
        assert len(device_ids) == 1, (
            'Currently, DistributedDataParallelWrapper only supports one'
            'single CUDA device for each process.'
            f'The length of device_ids must be 1, but got {len(device_ids)}.')
        self.module = module
        self.dim = dim
        self.to_ddp(
            device_ids=device_ids,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            **kwargs)

    def to_ddp(self, device_ids, dim, broadcast_buffers,
               find_unused_parameters, **kwargs):
        """Wrap models with separate MMDistributedDataParallel.

        It only wraps the modules with parameters.
        """
        for name, module in self.module._modules.items():
            if next(module.parameters(), None) is None:
                module = module.cuda()
            elif all(not p.requires_grad for p in module.parameters()):
                module = module.cuda()
            else:
                module = MMDistributedDataParallel(
                    module.cuda(),
                    device_ids=device_ids,
                    dim=dim,
                    broadcast_buffers=broadcast_buffers,
                    find_unused_parameters=find_unused_parameters,
                    **kwargs)
            self.module._modules[name] = module

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        This method is modified from :meth:`torch.nn.Module.state_dict`:
        1. Support getting sub-module for MMDataParallel or
            MMDistributedDataParallel.

        Args:
            destination (OrderedDict): Returned dict for the state of the
                module.
            prefix (str): Prefix of the key.
            keep_vars (bool): Whether to keep the variable property of the
                parameters. Default: False.

        Returns:
            dict: a dictionary containing a whole state of the module.
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(
            version=self.module._version)
        for name, param in self.module._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self.module._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data
        for name, module in self.module._modules.items():
            if module is not None:
                # this is what we modified: if sub-module is wrapped by
                # DataParallel or DistributedDataParallel, get its module
                if isinstance(module, (DataParallel, DistributedDataParallel)):
                    module = module.module
                module.state_dict(
                    destination, prefix + name + '.', keep_vars=keep_vars)
        for hook in module._state_dict_hooks.values():
            hook_result = hook(module, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])

    def train_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output

    def val_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        output = self.module.val_step(*inputs[0], **kwargs[0])
        return output
