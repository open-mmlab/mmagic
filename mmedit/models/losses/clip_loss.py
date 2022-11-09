# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmedit.registry import MODULES
from mmedit.utils import try_import

clip = try_import('clip')


class CLIPLossModel(torch.nn.Module):
    """Wrapped clip model to calculate clip loss.

    Ref: https://github.com/orpatashnik/StyleCLIP/blob/main/criteria/clip_loss.py # noqa

    Args:
        in_size (int, optional): Input image size. Defaults to 1024.
        scale_factor (int, optional): Unsampling factor. Defaults to 7.
        pool_size (int, optional): Pooling output size. Defaults to 224.
        clip_type (str, optional): A model name listed by
            `clip.available_models()`, or the path to a model checkpoint
            containing the state_dict. For more details, you can refer to
            https://github.com/openai/CLIP/blob/573315e83f07b53a61ff5098757e8fc885f1703e/clip/clip.py#L91 # noqa
            Defaults to 'ViT-B/32'.
    """

    def __init__(self,
                 in_size=1024,
                 scale_factor=7,
                 pool_size=224,
                 clip_type='ViT-B/32'):
        super(CLIPLossModel, self).__init__()
        try:
            import clip
        except ImportError:
            raise 'To use clip loss, openai clip need to be installed first'

        assert clip is not None, (
            "Cannot import 'clip'. Please install 'clip' via "
            "\"pip install git+https://github.com/openai/CLIP.git\".")
        self.model, self.preprocess = clip.load(clip_type, device='cpu')
        self.upsample = torch.nn.Upsample(scale_factor=scale_factor)
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=(scale_factor * in_size // pool_size))

    def forward(self, image=None, text=None):
        """Forward function."""
        assert image is not None
        assert text is not None
        image = self.avg_pool(self.upsample(image))
        loss = 1 - self.model(image, text)[0] / 100
        return loss


@MODULES.register_module()
class CLIPLoss(nn.Module):
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

        super(CLIPLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_info = data_info
        self.net = CLIPLossModel(**clip_model)
        self._loss_name = loss_name

    def forward(self, image, text):
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
        return self.net(image, text) * self.loss_weight
