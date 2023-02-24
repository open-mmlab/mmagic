# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from mmengine.config import ConfigDict
from mmengine.structures import BaseDataElement
from torch import Tensor

ForwardInputs = Tuple[Dict[str, Union[Tensor, str, int]], Tensor]
SampleList = Sequence[BaseDataElement]

NoiseVar = Union[Tensor, Callable, None]
LabelVar = Union[Tensor, Callable, List[int], None]

ConfigType = Union[ConfigDict, Dict]

InputType = Union[str, int, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[BaseDataElement, SampleList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], BaseDataElement, List[BaseDataElement]]
