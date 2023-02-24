# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional, Union

import torch

from mmedit.utils import (ConfigType, create_device, register_all_modules,
                          set_random_seed)
from ..inferencer_utils import BaseMMEditInferencer


class DiffusersInferencer(BaseMMEditInferencer):

    def __init__(self,
                 config: Union[ConfigType, str],
                 ckpt: Optional[str] = None,
                 device: Optional[str] = None,
                 extra_parameters: Optional[Dict] = None,
                 seed: int = 2022,
                 **kwargs) -> None:
        # Load config to cfg
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        register_all_modules()
        super().__init__(config, ckpt, device)

        self._init_extra_parameters(extra_parameters)
        self.base_params = self._dispatch_kwargs(**kwargs)
        self.seed = seed
        set_random_seed(self.seed)

        # from modelscope
        self.device_name = device
        self.cfg = None
        self.preprocessor = None
        self.framework = None
        self.device = create_device(self.device_name)
        # make sure we download the model from modelscope hub
        # model_folder = model

        # self.model = model_folder
        self.models = [self.model]
        self.has_multiple_models = len(self.models) > 1

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
