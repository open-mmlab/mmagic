# Copyright (c) OpenMMLab. All rights reserved.
import re
import sys
import warnings


def modify_args():
    for i, v in enumerate(sys.argv):
        if i == 0:
            assert v.endswith('.py')
        elif re.match(r'--\w+_.*', v):
            new_arg = v.replace('_', '-')
            warnings.warn(
                f'command line argument {v} is deprecated, '
                f'please use {new_arg} instead.',
                category=DeprecationWarning,
            )
            sys.argv[i] = new_arg
