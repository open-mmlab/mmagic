# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Sequence, Tuple, Union

from mmengine.config import ConfigDict
from mmengine.structures import BaseDataElement
from torch import Tensor

ForwardInputs = Tuple[Dict[str, Union[Tensor, str, int]], Tensor]
SampleList = Sequence[BaseDataElement]

NoiseVar = Union[Tensor, Callable, None]
LabelVar = Union[Tensor, Callable, List[int], None]

ConfigType = Union[ConfigDict, Dict]
