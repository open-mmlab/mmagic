# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmedit.registry import MODULES
from ..clip_loss import CLIPLossModel


@MODULES.register_module()
class CLIPLossComps(nn.Module):
    """Clip loss. In styleclip, this loss is used to optimize the latent code
    to generate image that match the text.

    In this loss, we may need to provide ``image``, ``text``. Thus,
    an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            image='fake_imgs',
            text='descriptions')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        clip_model (dict, optional): Kwargs for clip loss model. Defaults to
            dict().
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_clip'.
    """

    def __init__(self,
                 loss_weight=1.0,
                 data_info=None,
                 clip_model=dict(),
                 loss_name='loss_clip'):

        super().__init__()
        self.loss_weight = loss_weight
        self.data_info = data_info
        self.net = CLIPLossModel(**clip_model)
        self._loss_name = loss_name

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function,
        ``third_party_net_loss``.
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
        return self.net(*args, **kwargs) * self.loss_weight

    @staticmethod
    def loss_name():
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return 'clip_loss'
