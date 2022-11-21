# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

from .version import __version__, version_info

try:
    from mmcv.utils import digit_version
except ImportError:

    def digit_version(version_str):
        digit_ver = []
        for x in version_str.split('.'):
            if x.isdigit():
                digit_ver.append(int(x))
            elif x.find('rc') != -1:
                patch_version = x.split('rc')
                digit_ver.append(int(patch_version[0]) - 1)
                digit_ver.append(int(patch_version[1]))
        return digit_ver


MMCV_MIN = '1.3.13'
MMCV_MAX = '1.8'

mmcv_min_version = digit_version(MMCV_MIN)
mmcv_max_version = digit_version(MMCV_MAX)
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_min_version <= mmcv_version < mmcv_max_version), \
    f'mmcv=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv-full>={mmcv_min_version}, <={mmcv_max_version}.'

__all__ = ['__version__', 'version_info']
