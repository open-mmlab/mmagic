# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LoRACheckpointToSaveHook(Hook):
    """Pick up LoRA weights from checkpoint.

    Args:
        lora_keys (List[str]):
    """
    priority = 'VERY_LOW'

    def __init__(self, lora_keys: List[str] = ['lora_mapping']):
        super().__init__()
        self.lora_keys = lora_keys

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """
        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """

        new_ckpt = OrderedDict()
        for k in checkpoint['state_dict'].keys():
            if any(key in k for key in self.lora_keys):
                new_ckpt[k] = checkpoint['state_dict'][k]

        checkpoint['state_dict'] = new_ckpt
