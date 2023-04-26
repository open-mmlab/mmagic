# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Callable, List, Optional, Union

from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend, list_from_file

from ..registry import DATASETS


@DATASETS.register_module()
class BasicFramesDataset(BaseDataset):
    """BasicFramesDataset for open source projects in OpenMMLab/MMagic.

    This dataset is designed for low-level vision tasks with frames,
    such as video super-resolution and video frame interpolation.

    The annotation file is optional.

    If use annotation file, the annotation format can be shown as follows.

    .. code-block:: none

        Case 1 (Vid4):

            calendar 41
            city 34
            foliage 49
            walk 47

        Case 2 (REDS):

            000/00000000.png (720, 1280, 3)
            000/00000001.png (720, 1280, 3)

        Case 3 (Vimeo90k):

            00001/0266 (256, 448, 3)
            00001/0268 (256, 448, 3)

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img='', gt='').
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
        search_key (str): The key used for searching the folder to get
            data_list. Default: 'gt'.
        backend_args (dict, optional): Arguments to instantiate the prefix of
            uri corresponding backend. Defaults to None.
        depth (int): The depth of path. Default: 1
        num_input_frames (None | int): Number of input frames. Default: None.
        num_output_frames (None | int): Number of output frames. Default: None.
        fixed_seq_len (None | int): The fixed sequence length.
            If None, BasicFramesDataset will obtain the length of each
            sequence.
            Default: None.
        load_frames_list (dict): Load frames list for each key.
            Default: dict().

    Examples:

        Assume the file structure as the following:

        mmagic (root)
        ├── mmagic
        ├── tools
        ├── configs
        ├── data
        │   ├── Vid4
        │   │   ├── BIx4
        │   │   │   ├── city
        │   │   │   │   ├── img1.png
        │   │   ├── GT
        │   │   │   ├── city
        │   │   │   │   ├── img1.png
        │   │   ├── meta_info_Vid4_GT.txt
        │   ├── places
        │   │   ├── sequences
        |   |   |   ├── 00001
        │   │   │   │   ├── 0389
        │   │   │   │   │   ├── img1.png
        │   │   │   │   │   ├── img2.png
        │   │   │   │   │   ├── img3.png
        │   │   ├── tri_trainlist.txt

        Case 1: Loading Vid4 dataset for training a VSR model.

        .. code-block:: python

            dataset = BasicFramesDataset(
                ann_file='meta_info_Vid4_GT.txt',
                metainfo=dict(dataset_type='vid4', task_name='vsr'),
                data_root='data/Vid4',
                data_prefix=dict(img='BIx4', gt='GT'),
                pipeline=[],
                depth=2,
                num_input_frames=5)

        Case 2: Loading Vimeo90k dataset for training a VFI model.

        .. code-block:: python

            dataset = BasicFramesDataset(
                ann_file='tri_trainlist.txt',
                metainfo=dict(dataset_type='vimeo90k', task_name='vfi'),
                data_root='data/vimeo-triplet',
                data_prefix=dict(img='sequences', gt='sequences'),
                pipeline=[],
                depth=2,
                load_frames_list=dict(
                    img=['img1.png', 'img3.png'], gt=['img2.png']))

        See more details in unittest
            tests/test_datasets/test_base_frames_dataset.py
                TestFramesDatasets().test_version_1_method()
    """

    METAINFO = dict(dataset_type='base_edit_dataset', task_name='editing')

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 filename_tmpl: dict = dict(),
                 search_key: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 depth: int = 1,
                 num_input_frames: Optional[int] = None,
                 num_output_frames: Optional[int] = None,
                 fixed_seq_len: Optional[int] = None,
                 load_frames_list: dict = dict(),
                 **kwargs):

        for key in data_prefix:
            if key not in filename_tmpl:
                filename_tmpl[key] = '{}'

        if search_key is None:
            keys = list(data_prefix.keys())
            search_key = keys[0]
        self.search_key = search_key
        self.filename_tmpl = filename_tmpl
        self.use_ann_file = (ann_file != '')
        if backend_args is None:
            self.backend_args = None
        else:
            self.backend_args = backend_args.copy()
        self.depth = depth
        self.seq_lens = dict(fixed_seq_len=fixed_seq_len)
        self.num_input_frames = num_input_frames
        self.num_output_frames = num_output_frames
        self.load_frames_list = load_frames_list
        self.file_backend = get_file_backend(
            uri=data_root, backend_args=backend_args)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load data list from folder or annotation file.

        Returns:
            list[dict]: A list of annotation.
        """

        path_list = self._get_path_list()
        self._set_seq_lens()

        data_list = []
        for path in path_list:
            basename, _ = osp.splitext(path)
            sequence_length = self.seq_lens['fixed_seq_len']
            if sequence_length is None:
                sequence_length = self.seq_lens[path.split(os.sep)[0]]
            data = dict(
                key=basename,
                num_input_frames=self.num_input_frames,
                num_output_frames=self.num_output_frames,
                sequence_length=sequence_length)
            for key in self.data_prefix:
                if key in self.load_frames_list:
                    folder = osp.join(self.data_prefix[key], path)
                    data[f'{key}_path'] = self._get_frames_list(key, folder)
                    # The list of frames has been loaded,
                    # ``sequence_length`` is useless
                    # Avoid loading frames by ``sequence_length`` in pipeline
                    data['sequence_length'] = None
                    # overwrite ``num_input_frames`` and ``num_output_frames``
                    if key == 'img':
                        data['num_input_frames'] = len(data[f'{key}_path'])
                    elif key == 'gt':
                        data['num_output_frames'] = len(data[f'{key}_path'])
                else:
                    data[f'{key}_path'] = self.data_prefix[key]
            data_list.append(data)

        return data_list

    def _get_path_list(self):
        """Get list of paths from annotation file or folder of dataset.

        Returns:
            list[str]: A list of paths.
        """

        if self.use_ann_file:
            path_list = self._get_path_list_from_ann()
        else:
            path_list = self._get_path_list_from_folder(depth=self.depth)

        return path_list

    def _get_path_list_from_ann(self):
        """Get list of paths from annotation file.

        Returns:
            list[str]: A list of paths.
        """

        ann_list = list_from_file(
            self.ann_file, backend_args=self.backend_args)
        path_list = []
        for ann in ann_list:
            if ann.isspace() or ann == '':
                continue
            path = ann.split(' ')[0]
            # Compatible with Windows file systems
            path = path.replace('/', os.sep)
            splitted_path = path.split(os.sep)
            if self.seq_lens['fixed_seq_len'] is None:
                self.seq_lens[splitted_path[0]] = 0
            ann_depth = len(splitted_path)
            if self.depth > ann_depth:
                # desire "folder/file", but the ann_file provides "folder".
                sub_path_list = self._get_path_list_from_folder(
                    sub_folder=path,
                    need_ext=False,
                    depth=self.depth - ann_depth)
                path_list.extend(
                    [path + os.sep + sub_path for sub_path in sub_path_list])
            elif self.depth < ann_depth:
                # desire "folder", while the ann_file provides "folder/file".
                desire_path = f'{os.sep}'.join(splitted_path[:self.depth])
                if desire_path not in path_list:
                    path_list.append(desire_path)
            else:
                # desire "folder/file" and the ann_file provides "folder/file".
                # or desire "folder" and the ann_file provides "folder".
                path_list.append(path)

        return path_list

    def _get_path_list_from_folder(self,
                                   sub_folder=None,
                                   need_ext=True,
                                   depth=1):
        """Get list of paths from folder.

        Args:
            sub_folder (None | str): The path of sub_folder. Default: None.
            need_ext (bool): Whether need ext. Default: True.
            depth (int): Residual depth of path, recursively called to
                ``depth == 1``. Default: 1

        Returns:
            list[str]: A list of paths.
        """

        folder = self.data_prefix[self.search_key]
        tmpl = self.filename_tmpl[self.search_key].format('')
        path_list = []
        if sub_folder:
            folder = osp.join(folder, sub_folder)
        listdir = list(self.file_backend.list_dir_or_file(dir_path=folder))
        listdir.sort()
        for path in listdir:
            basename, ext = osp.splitext(path)
            if not (sub_folder or self.seq_lens['fixed_seq_len']):
                self.seq_lens[basename] = 0
            if depth > 1:
                sub_path_list = self._get_path_list_from_folder(
                    sub_folder=path)
                path_list.extend(
                    [path + os.sep + sub_path for sub_path in sub_path_list])
            elif basename.endswith(tmpl):
                if need_ext:
                    path = path.replace(tmpl + ext, ext)
                else:
                    path = path.replace(tmpl + ext, '')
                path_list.append(path)

        return path_list

    def _set_seq_lens(self):
        """Get sequence lengths."""

        if self.seq_lens['fixed_seq_len']:
            return
        folder = self.data_prefix[self.search_key]
        for key in self.seq_lens.keys():
            if key == 'fixed_seq_len':
                continue
            path = osp.join(folder, key)
            num_frames = len(list(self.file_backend.list_dir_or_file(path)))
            self.seq_lens[key] = num_frames

    def _get_frames_list(self, key, folder):
        """Obtain list of frames.

        Args:
            key (str): The key of frames list, e.g. ``img``, ``gt``.
            folder (str): Folder of frames.

        Return:
            list[str]: The paths list of frames.
        """

        if 'all' in self.load_frames_list[key]:
            # load all
            files = list(self.file_backend.list_dir_or_file(dir_path=folder))
        else:
            files = self.load_frames_list[key]

        files.sort()
        tmpl = self.filename_tmpl[key]
        files = [tmpl.format(file) for file in files]
        paths = [osp.join(folder, file) for file in files]

        return paths
