# Copyright (c) Open-MMLab. All rights reserved.

__version__ = '0.12.0'


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


def _minimal_ext_cmd(cmd):
    import os
    import subprocess

    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out


def get_public_or_local_version(force_public=False):
    """Get PEP440 compatible public or local version identifier.

    Returns:
        str: PEP440 compatible version identifier.
             public ones are like 0.12.0
             private ones are like 0.12.0+ge9f4097, g means git
    """
    import re

    try:
        git_output = _minimal_ext_cmd(['git', 'describe', '--tags']).decode()
        # will output v0.12.0-35-ge9f4097 when not on tags
        # and v0.12.0 when on tags
    except OSError:
        return __version__

    public_match = re.match(r'v([\d+](?:\.\d+)+)$', git_output)
    local_match = re.match(r'v([\d+](?:\.\d+)+)-\d+-(\w+)', git_output)
    assert public_match or local_match

    if public_match or force_public:
        version = public_match.group(1)
    elif local_match:
        version = F'{local_match.group(1)}+{local_match.group(2)}'

    return version


def get_local_version_with_git_hash(fallback_to_public=True):
    """Get PEP440 compatible local version identifier.

    Returns:
        str: Local version identifier, like 0.12.0+ge9f4097, g means git
    """
    import re

    try:
        git_output = _minimal_ext_cmd(['git', 'describe', '--tags',
                                       '--long']).decode()
        # will output v0.12.0-35-ge9f4097 when not on tags
        # and v0.12.0-0-gbf53426 when on tags
    except OSError as e:
        if fallback_to_public:
            return __version__
        else:
            raise e

    local_match = re.match(r'v([\d+](?:\.\d+)+)-\d+-(\w+)', git_output)

    if local_match:
        version = F'{local_match.group(1)}+{local_match.group(2)}'
    else:
        raise RuntimeError

    return version


__local_version__ = get_local_version_with_git_hash()
