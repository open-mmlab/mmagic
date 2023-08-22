# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmengine

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


<<<<<<< HEAD:mmagic/__init__.py
MMCV_MIN = '2.0.0'
MMCV_MAX = '2.1.0'
=======
MMCV_MIN = '1.3.13'
MMCV_MAX = '1.8'

>>>>>>> 6f2f3ae2ad3e365f94bbf19c01a1d1056dad3895:mmedit/__init__.py
mmcv_min_version = digit_version(MMCV_MIN)
mmcv_max_version = digit_version(MMCV_MAX)
mmcv_version = digit_version(mmcv.__version__)

MMENGINE_MIN = '0.4.0'
MMENGINE_MAX = '1.0.0'
mmengine_min_version = digit_version(MMENGINE_MIN)
mmengine_max_version = digit_version(MMENGINE_MAX)
mmengine_version = digit_version(mmengine.__version__)

assert (mmcv_min_version <= mmcv_version < mmcv_max_version), \
    f'mmcv=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv-full>={mmcv_min_version}, <{mmcv_max_version}.'

assert (mmengine_min_version <= mmengine_version < mmengine_max_version), \
    f'mmengine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_min_version}, ' \
    f'<{mmengine_max_version}.'

__all__ = ['__version__', 'version_info']
