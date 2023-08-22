# Copyright (c) Open-MMLab. All rights reserved.

<<<<<<< HEAD:mmagic/version.py
__version__ = '1.0.2dev0'
=======
__version__ = '0.16.1'
>>>>>>> 6f2f3ae2ad3e365f94bbf19c01a1d1056dad3895:mmedit/version.py


def parse_version_info(version_str):
    ver_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            ver_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            ver_info.append(int(patch_version[0]))
            ver_info.append(f'rc{patch_version[1]}')
    return tuple(ver_info)


version_info = parse_version_info(__version__)
