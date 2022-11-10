# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722.

    Note: Important!
    This MoCo model does not use batch_shuffle_ddp() and concat_all_gather()

    TODO: Explore the impacts of batch_shuffle_ddp()

    Args:
        base_encoder (nn.Module): base model for both q and k encoders.
        K (int): Number of negative keys maintained in the
            queue.
            Defaults to 65536.
        dim (int): Dimension of compact feature vectors.
            Defaults to 128.
        m (float): Momentum coefficient for the momentum-updated
            encoder.
            Defaults to 0.999.
        T (float): Softmax temperature.
            Default: 0.07.
    """

    def __init__(self, base_encoder, dim=128, K=3 * 256, m=0.999, T=0.07):

        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = AllGatherLayer.forward(ctx, keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """Forward function.

        Args:
            im_q: a batch of query images
            im_k: a batch of key images

        Return:
            if training:
                feat, logits, labels, inter
            else
                feat, inter
        """
        # compute query features
        embedding, q, inter = self.encoder_q(im_q)  # queries: NxC

        # outputs dictionary
        outputs = dict(feat=embedding, inter=inter)

        if self.training:
            # normalizing q
            q = nn.functional.normalize(q, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                _, k, _ = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            outputs['logits'] = logits
            outputs['labels'] = labels

        return outputs
