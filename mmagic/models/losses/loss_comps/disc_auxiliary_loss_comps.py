# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn

from mmagic.registry import MODELS
from ..gan_loss import (disc_shift_loss, gradient_penalty_loss,
                        r1_gradient_penalty_loss)


@MODELS.register_module()
class DiscShiftLossComps(nn.Module):
    """Disc Shift Loss.

    This loss is proposed in PGGAN as an auxiliary loss for discriminator.

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
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size)

    But in this loss, we will need to provide ``pred`` as input. Thus, an
    example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            pred='disc_pred_fake')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    In addition, in general, ``disc_shift_loss`` will be applied over real and
    fake data. In this case, users just need to add this loss module twice, but
    with different ``data_info``. Our model will automatically add these two
    items.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_disc_shift'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 data_info: Optional[dict] = None,
                 loss_name: str = 'loss_disc_shift') -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.data_info = data_info
        self._loss_name = loss_name

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function, ``disc_shift_loss``.
        """
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
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            return disc_shift_loss(**kwargs) * self.loss_weight
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return disc_shift_loss(*args, **kwargs) * self.loss_weight

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


@MODELS.register_module()
class GradientPenaltyLossComps(nn.Module):
    """Gradient Penalty for WGAN-GP.

    In the detailed implementation, there are two streams where one uses the
    pixel-wise gradient norm, but the other adopts normalization along instance
    (HWC) dimensions. Thus, ``norm_mode`` are offered to define which mode you
    want.

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
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size)

    But in this loss, we will need to provide ``discriminator``, ``real_data``,
    and ``fake_data`` as input. Thus, an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            discriminator='disc',
            real_data='real_imgs',
            fake_data='fake_imgs')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        norm_mode (str): This argument decides along which dimension the norm
            of the gradients will be calculated. Currently, we support ["pixel"
            , "HWC"]. Defaults to "pixel".
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_gp'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 norm_mode: str = 'pixel',
                 data_info: Optional[dict] = None,
                 loss_name: str = 'loss_gp') -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.norm_mode = norm_mode
        self.data_info = data_info
        self._loss_name = loss_name

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function,
        ``gradient_penalty_loss``.
        """
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
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            # kwargs.update(
            #     dict(weight=self.loss_weight, norm_mode=self.norm_mode))
            kwargs.update(dict(norm_mode=self.norm_mode))
            return gradient_penalty_loss(**kwargs) * self.loss_weight
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            # return gradient_penalty_loss(
            #     *args, weight=self.loss_weight, **kwargs)
            return gradient_penalty_loss(*args, **kwargs) * self.loss_weight

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


@MODELS.register_module()
class R1GradientPenaltyComps(nn.Module):
    """R1 gradient penalty for WGAN-GP.

    R1 regularizer comes from:
    "Which Training Methods for GANs do actually Converge?" ICML'2018

    Different from original gradient penalty, this regularizer only penalized
    gradient w.r.t. real data.

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
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size)

    But in this loss, we will need to provide ``discriminator`` and
    ``real_data`` as input. Thus, an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            discriminator='disc',
            real_data='real_imgs')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        norm_mode (str): This argument decides along which dimension the norm
            of the gradients will be calculated. Currently, we support ["pixel"
            , "HWC"]. Defaults to "pixel".
        interval (int, optional): The interval of calculating this loss.
            Defaults to 1.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_r1_gp'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 norm_mode: str = 'pixel',
                 interval: int = 1,
                 data_info: Optional[dict] = None,
                 use_apex_amp: bool = False,
                 loss_name: str = 'loss_r1_gp') -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.norm_mode = norm_mode
        self.interval = interval
        self.data_info = data_info
        self.use_apex_amp = use_apex_amp
        self._loss_name = loss_name

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function,
        ``r1_gradient_penalty_loss``.
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
                dict(norm_mode=self.norm_mode, use_apex_amp=self.use_apex_amp))
            return r1_gradient_penalty_loss(**kwargs) * self.loss_weight
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return r1_gradient_penalty_loss(
                *args, norm_mode=self.norm_mode, **kwargs) * self.loss_weight

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
