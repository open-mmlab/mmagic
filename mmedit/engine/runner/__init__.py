# Copyright (c) OpenMMLab. All rights reserved.
from .edit_loops import EditTestLoop, EditValLoop
from .gen_loops import GenTestLoop, GenValLoop
from .log_processor import GenLogProcessor
from .multi_loops import MultiTestLoop, MultiValLoop

__all__ = [
    'EditTestLoop', 'EditValLoop', 'MultiValLoop', 'MultiTestLoop',
    'GenTestLoop', 'GenValLoop', 'GenLogProcessor'
]
