# Copyright (c) OpenMMLab. All rights reserved.
import gzip
import hashlib
import os
import os.path
import os.path as osp
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from os import PathLike
from typing import Callable, Dict, List, Tuple

from mmengine.fileio.backends import BaseStorageBackend


# TODO: we can use FileClient.infer_client to replace this function
def infer_io_backend(data_root: str) -> str:
    """Infer the io backend from the given data_root.

    Args:
        data_root (str): The path of data root.

    Returns:
        str: The io backend.
    """
    if (data_root.upper().startswith('HTTP')
            or data_root.upper().startswith('HTTPS')):
        backend = 'http'
    elif data_root.upper().startswith('S3') or (
            len(data_root.split(':')) > 2
            and data_root.split(':')[1].upper() == 'S3'):
        # two case:
        # 1. s3://xxxxx (raw petrel path)
        # 2. CONFIG:s3://xxx  (petrel path with specific config)
        backend = 'petrel'
    else:
        # use default one
        backend = 'local'
    return backend


def calculate_md5(fpath: str,
                  file_backend: BaseStorageBackend = None,
                  chunk_size: int = 1024 * 1024) -> str:
    """Calculate MD5 of the file.

    Args:
        fpath (str): The path of the file.
        file_backend (BaseStorageBackend, optional): The file backend to fetch
            the file. Defaults to None.
        chunk_size (int, optional): The chunk size to calculate MD5. Defaults
            to 1024*1024.

    Returns:
        str: The string of MD5.
    """
    md5 = hashlib.md5()
    if file_backend is None or file_backend.name == 'LocalBackend':
        with open(fpath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
    else:
        md5.update(file_backend.get(fpath))
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs) -> bool:
    """Checn whether the MD5 of the file.

    Args:
        fpath (str): The path of the file.
        md5 (str): Target MD5 value.

    Returns:
        bool: If true, the MD5 of passed file is same as target MD5.
    """
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None) -> bool:
    """Check whether the file is integrity by comparing the MD5 of the file
    with target MD5.

    Args:
        fpath (str): The path of the file.
        md5 (str, optional): The target MD5 value. Defaults to None.

    Returns:
        bool: If true, the passed file is integrity.
    """
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    """Download object at the given URL to a local path.

    Modified from
    https://pytorch.org/docs/stable/hub.html#torch.hub.download_url_to_file

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved,
            e.g. ``/tmp/temporary_file``
        hash_prefix (string, optional): If not None, the SHA256 downloaded
            file should start with ``hash_prefix``. Defaults to None.
        progress (bool): whether or not to display a progress bar to stderr.
            Defaults to True
    """
    file_size = None
    req = urllib.request.Request(url)
    u = urllib.request.urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders('Content-Length')
    else:
        content_length = meta.get_all('Content-Length')
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after download is
    # complete. This prevents a local file being overridden by a broken
    # download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    import rich.progress
    columns = [
        rich.progress.DownloadColumn(),
        rich.progress.BarColumn(bar_width=None),
        rich.progress.TimeRemainingColumn(),
    ]
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with rich.progress.Progress(*columns) as pbar:
            task = pbar.add_task('download', total=file_size, visible=progress)
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(task, advance=len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from.
        root (str): Directory to place downloaded file in.
        filename (str | None): Name to save the file under.
            If filename is None, use the basename of the URL.
        md5 (str | None): MD5 checksum of the download.
            If md5 is None, download without md5 check.
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print(f'Using downloaded and verified file: {fpath}')
    else:
        try:
            print(f'Downloading {url} to {fpath}')
            download_url_to_file(url, fpath)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      f' Downloading {url} to {fpath}')
                download_url_to_file(url, fpath)
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError('File not found or corrupted.')


def _is_tarxz(filename):
    """Judge whether the file is `.tar.xz`"""
    return filename.endswith('.tar.xz')


def _is_tar(filename):
    """Judge whether the file is `.tar`"""
    return filename.endswith('.tar')


def _is_targz(filename):
    """Judge whether the file is `.tar.gz`"""
    return filename.endswith('.tar.gz')


def _is_tgz(filename):
    """Judge whether the file is `.tgz`"""
    return filename.endswith('.tgz')


def _is_gzip(filename):
    """Judge whether the file is `.gzip`"""
    return filename.endswith('.gz') and not filename.endswith('.tar.gz')


def _is_zip(filename):
    """Judge whether the file is `.zip`"""
    return filename.endswith('.zip')


def extract_archive(from_path, to_path=None, remove_finished=False):
    """Extract the archive."""
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(
            to_path,
            os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, 'wb') as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError(f'Extraction of {from_path} not supported')

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(url,
                                 download_root,
                                 extract_root=None,
                                 filename=None,
                                 md5=None,
                                 remove_finished=False):
    """Download and extract the archive."""
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f'Extracting {archive} to {extract_root}')
    extract_archive(archive, extract_root, remove_finished)


def open_maybe_compressed_file(path: str):
    """Return a file object that possibly decompresses 'path' on the fly.

    Decompression occurs when argument `path` is a string and ends with '.gz'
    or '.xz'.
    """
    if not isinstance(path, str):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def expanduser(path):
    """Expand ~ and ~user constructions.

    If user or $HOME is unknown, do nothing.
    """
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


def find_folders(root: str, file_backend: BaseStorageBackend
                 ) -> Tuple[List[str], Dict[str, int]]:
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        Tuple[List[str], Dict[str, int]]:

        - folders: The name of sub folders under the root.
        - folder_to_idx: The map from folder name to class idx.
    """
    folders = list(
        file_backend.list_dir_or_file(
            root,
            list_dir=True,
            list_file=False,
            recursive=False,
        ))
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folders, folder_to_idx


def get_samples(root: str, folder_to_idx: Dict[str, int],
                is_valid_file: Callable, file_backend: BaseStorageBackend):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        is_valid_file (Callable): A function that takes path of a file
            and check if the file is a valid sample file.

    Returns:
        Tuple[list, set]:

        - samples: a list of tuple where each element is (image, class_idx)
        - empty_folders: The folders don't have any valid files.
    """
    samples = []
    available_classes = set()

    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = file_backend.join_path(root, folder_name)
        files = list(
            file_backend.list_dir_or_file(
                _dir,
                list_dir=False,
                list_file=True,
                recursive=True,
            ))
        for file in sorted(list(files)):
            if is_valid_file(file):
                path = file_backend.join_path(folder_name, file)
                item = (path, folder_to_idx[folder_name])
                samples.append(item)
                available_classes.add(folder_name)

    empty_folders = set(folder_to_idx.keys()) - available_classes

    return samples, empty_folders
