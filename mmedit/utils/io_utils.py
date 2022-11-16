# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os

import click
import mmengine
import requests
import torch.distributed as dist
from mmengine.dist import get_dist_info
from requests.exceptions import InvalidURL, RequestException, Timeout

MMEDIT_CACHE_DIR = os.path.expanduser('~') + '/.cache/openmmlab/mmedit/'


def get_content_from_url(url, timeout=15, stream=False):
    """Get content from url.

    Args:
        url (str): Url for getting content.
        timeout (int): Set the socket timeout. Default: 15.
    """
    try:
        response = requests.get(url, timeout=timeout, stream=stream)
    except InvalidURL as err:
        raise err  # type: ignore
    except Timeout as err:
        raise err  # type: ignore
    except RequestException as err:
        raise err  # type: ignore
    except Exception as err:
        raise err  # type: ignore
    return response


def download_from_url(url,
                      dest_path=None,
                      dest_dir=MMEDIT_CACHE_DIR,
                      hash_prefix=None):
    """Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download.
        dest_path (str): Path where object will be saved.
        dest_dir (str): The directory of the destination. Defaults to
            ``'~/.cache/openmmlab/mmgen/'``.
        hash_prefix (string, optional): If not None, the SHA256 downloaded
            file should start with `hash_prefix`. Default: None.

    Return:
        str: path for the downloaded file.
    """
    # get the exact destination path
    if dest_path is None:
        filename = url.split('/')[-1]
        dest_path = os.path.join(dest_dir, filename)

    if dest_path.startswith('~'):
        dest_path = os.path.expanduser('~') + dest_path[1:]

    # advoid downloading existed file
    if os.path.exists(dest_path):
        return dest_path

    rank, ws = get_dist_info()

    # only download from the master process
    if rank == 0:
        # mkdir
        _dir = os.path.dirname(dest_path)
        mmengine.mkdir_or_exist(_dir)

        if hash_prefix is not None:
            sha256 = hashlib.sha256()

        response = get_content_from_url(url, stream=True)
        size = int(response.headers.get('content-length'))
        with open(dest_path, 'wb') as fw:
            content_iter = response.iter_content(chunk_size=1024)
            with click.progressbar(content_iter, length=size / 1024) as chunks:
                for chunk in chunks:
                    if chunk:
                        fw.write(chunk)
                        fw.flush()
                        if hash_prefix is not None:
                            sha256.update(chunk)

        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    f'invalid hash value, expected "{hash_prefix}", but got '
                    f'"{digest}"')

    # sync the other processes
    if ws > 1:
        dist.barrier()

    return dest_path
