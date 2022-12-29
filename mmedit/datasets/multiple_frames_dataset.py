# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from ..registry import DATASETS
from .basic_frames_dataset import BasicFramesDataset


@DATASETS.register_module()
class MultipleFramesDataset(BasicFramesDataset):
    """MultipleFramesDataset for open source projects in OpenMMLab/MMEditing.

    This dataset is designed for low-level vision tasks with frames,
    especially for tasks containing video frame interpolation.

    Note: This dataset is mainly used for Multiple Frame Interpolation.

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
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Default: None.
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

        mmediting (root)
        ├── mmedit
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

        Case 1: Loading dataset for training a VFI model.

        .. code-block:: python

            dataset = BasicFramesDataset(
                ann_file='tri_trainlist.txt',
                metainfo=dict(dataset_type='vimeo90k', task_name='vfi'),
                data_root='data/vimeo-triplet',
                data_prefix=dict(img='sequences', gt='sequences'),
                pipeline=[],
                depth=2,
                load_frames_list=dict(
                    img=[0, 2], gt=[1]))

        Case 1: Loading dataset for training a MFI model.

        .. code-block:: python

            dataset = BasicFramesDataset(
                ann_file='tri_trainlist.txt',
                metainfo=dict(dataset_type='gopro', task_name='mfi'),
                data_root='data/vimeo-triplet',
                data_prefix=dict(img='sequences', gt='sequences'),
                pipeline=[],
                depth=2,
                load_frames_list=dict(
                    img=[0, 8], gt=[1, 2, 3, 4, 5, 6, 7]))
    """

    METAINFO = dict(
        dataset_type='multiple_edit_dataset', task_name='interpolation')

    def load_data_list(self) -> List[dict]:
        """Load data list from folder or annotation file.

        Returns:
            list[dict]: A list of annotation.
        """
        self._get_path_list()
        self._set_seq_lens()

        folders = []
        path_list_dict = {}
        for f in self.seq_lens.keys():
            if f == 'fixed_seq_len':
                continue
            folders.append(f)
            path = osp.join(self.data_prefix[self.search_key], f)
            path_list_dict[f] = sorted(
                list(self.file_client.list_dir_or_file(path)))

        # Analyze num of input frames
        whole_list = []
        if self.num_input_frames is not None:
            whole_list = list(range(self.num_input_frames))
        elif self.load_frames_list:
            for img_list in self.load_frames_list:
                whole_list.extend(self.load_frames_list[img_list])
            whole_list = list(set(whole_list))
        else:
            raise AttributeError(
                'num_input_frames and load_frames_list should not be Empty'
                'at the same time')
        num_frames_each_data = len(whole_list)

        data_list = []
        for folder in folders:
            num_data = (self.seq_lens[folder] -
                        num_frames_each_data) // (num_frames_each_data - 1) + 1
            first_index = [
                i * (num_frames_each_data - 1) for i in range(num_data)
            ]
            for index in first_index:
                path = path_list_dict[folder][index]
                basename, _ = osp.splitext(path)
                sequence_length = self.seq_lens['fixed_seq_len']
                if sequence_length is None:
                    sequence_length = self.seq_lens[folder]
                data = dict(
                    key=folder + '_' + basename,
                    num_input_frames=self.num_input_frames,
                    num_output_frames=self.num_output_frames,
                    num_frame_this_data=num_frames_each_data,
                    sequence_length=sequence_length)

                for key in self.data_prefix:
                    folder_path = osp.join(self.data_prefix[key], folder)
                    if key in self.load_frames_list:
                        data[f'{key}_path'] = self._get_frames_list(
                            index, self.load_frames_list[key],
                            path_list_dict[folder], folder_path)
                    else:
                        data[f'{key}_path'] = self._get_frames_list(
                            index, whole_list, path_list_dict[folder],
                            folder_path)
                    # The list of frames has been loaded,
                    # ``sequence_length`` is useless
                    # Avoid loading frames by ``sequence_length`` in pipeline
                    data['sequence_length'] = None
                    # overwrite ``num_input_frames`` and ``num_output_frames``
                    if key == 'img':
                        data['num_input_frames'] = len(data[f'{key}_path'])
                    elif key == 'gt':
                        data['num_output_frames'] = len(data[f'{key}_path'])
                data_list.append(data)
        return data_list

    def _get_frames_list(self, start_idx, img_index_list, folder, folder_path):
        indices = [x + start_idx for x in img_index_list]
        paths = [osp.join(folder_path, folder[idx]) for idx in indices]
        return paths
