# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.autograd as autograd
import torch.distributed as dist


class AllGatherLayer(autograd.Function):
    """All gather layer with backward propagation path.

    Indeed, this module is to make ``dist.all_gather()`` in the backward graph.
    Such kind of operation has been widely used in Moco and other contrastive
    learning algorithms.
    """

    @staticmethod
    def forward(ctx, x):
        """Forward function."""
        ctx.save_for_backward(x)
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward function."""
        x, = ctx.saved_tensors
        grad_out = torch.zeros_like(x)
        grad_out = grad_outputs[dist.get_rank()]
        return grad_out
