# Copyright (c) OpenMMLab. All rights reserved.
from .gen_loops import GenTestLoop, GenValLoop
from .multi_loops import MultiTestLoop, MultiValLoop

__all__ = ['MultiValLoop', 'MultiTestLoop', 'GenTestLoop', 'GenValLoop']
