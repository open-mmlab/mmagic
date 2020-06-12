from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    """Base model.

    All models should subclass it.
    All subclass should overwrite:

        ``init_weights``, supporting to initialize models.

        ``forward_train``, supporting to forward when training.

        ``forward_test``, supporting to forward when testing.

        ``train_step``, supporting to train one step when training.
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward_train(self, imgs, labels):
        pass

    @abstractmethod
    def forward_test(self, imgs):
        pass

    def forward(self, imgs, labels, test_mode, **kwargs):
        if not test_mode:
            return self.forward_train(imgs, labels, **kwargs)
        else:
            return self.forward_test(imgs, **kwargs)

    @abstractmethod
    def train_step(self, data_batch, optimizer):
        pass

    def val_step(self, data_batch, **kwargs):
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def parse_losses(self, losses):
        """Parse losses dict for different loss variants.

        Args:
            losses (dict): Loss dict.

        Returns:
            loss (float): Sum of the total loss.
            log_vars (dict): loss dict for different variants.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()

        return loss, log_vars
