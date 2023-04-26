# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn

from mmagic.registry import MODELS
from ..gan_loss import gen_path_regularizer


@MODELS.register_module()
class GeneratorPathRegularizerComps(nn.Module):
    """Generator Path Regularizer.

    Path regularization is proposed in StyleGAN2, which can help the improve
    the continuity of the latent space. More details can be found in:
    Analyzing and Improving the Image Quality of StyleGAN, CVPR2020.

    Users can achieve lazy regularization by setting ``interval`` arguments
    here.

    **Note for the design of ``data_info``:**
    In ``MMagic``, almost all of loss modules contain the argument
    ``data_info``, which can be used for constructing the link between the
    input items (needed in loss calculation) and the data from the generative
    model. For example, in the training of GAN model, we will collect all of
    important data/modules into a dictionary:

    .. code-block:: python
        :caption: Code from StaticUnconditionalGAN, train_step
        :linenos:

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake_g,
            iteration=curr_iter,
            batch_size=batch_size)

    But in this loss, we will need to provide ``generator`` and ``num_batches``
    as input. Thus an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            generator='gen',
            num_batches='batch_size')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        pl_batch_shrink (int, optional): The factor of shrinking the batch size
            for saving GPU memory. Defaults to 1.
        decay (float, optional): Decay for moving average of mean path length.
            Defaults to 0.01.
        pl_batch_size (int | None, optional): The batch size in calculating
            generator path. Once this argument is set, the ``num_batches`` will
            be overridden with this argument and won't be affected by
            ``pl_batch_shrink``. Defaults to None.
        sync_mean_buffer (bool, optional): Whether to sync mean path length
            across all of GPUs. Defaults to False.
        interval (int, optional): The interval of calculating this loss. This
            argument is used to support lazy regularization. Defaults to 1.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_path_regular'.
    """

    def __init__(self,
                 loss_weight: float = 1.,
                 pl_batch_shrink: int = 1,
                 decay: float = 0.01,
                 pl_batch_size: Optional[int] = None,
                 sync_mean_buffer: bool = False,
                 interval: int = 1,
                 data_info: Optional[dict] = None,
                 use_apex_amp: bool = False,
                 loss_name: str = 'loss_path_regular') -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.decay = decay
        self.pl_batch_size = pl_batch_size
        self.sync_mean_buffer = sync_mean_buffer
        self.interval = interval
        self.data_info = data_info
        self.use_apex_amp = use_apex_amp
        self._loss_name = loss_name

        self.register_buffer('mean_path_length', torch.tensor(0.))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function,
        ``gen_path_regularizer``.
        """
        if self.interval > 1:
            assert self.data_info is not None
        # use data_info to build computational path
        if self.data_info is not None:
            # parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict), (
                    'You should offer a dictionary containing network outputs '
                    'for building up computational graph of this loss module.')
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0, (
                    'If the outputs dict is given in keyworded arguments, no'
                    ' further non-keyworded arguments should be offered.')
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments passed to this loss module.'
                    ' Please check the usage of this module')
            if self.interval > 1 and outputs_dict[
                    'iteration'] % self.interval != 0:
                return None
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            kwargs.update(
                dict(
                    # weight=self.loss_weight,
                    mean_path_length=self.mean_path_length,
                    pl_batch_shrink=self.pl_batch_shrink,
                    decay=self.decay,
                    use_apex_amp=self.use_apex_amp,
                    pl_batch_size=self.pl_batch_size,
                    sync_mean_buffer=self.sync_mean_buffer))
            path_penalty, self.mean_path_length, _ = gen_path_regularizer(
                **kwargs)
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            path_penalty, self.mean_path_length, _ = gen_path_regularizer(
                *args, **kwargs)
        return path_penalty * self.loss_weight

    def loss_name(self) -> str:
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
