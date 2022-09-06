# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRFolderRefDataset(BaseSRDataset):
    """General paired image folder dataset for reference-based image
    restoration.

    The dataset loads ref (reference) image pairs
        Must contain: ref (reference)
        Optional: GT (Ground-Truth), LQ (Low Quality), or both
            Cannot only contain ref.

    Applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "folder mode", which needs to specify the ref folder path and
    gt folder path, each folder containing the corresponding images.
    Image lists will be generated automatically. You can also specify the
    filename template to match the image pairs.

    For example, we have three folders with the following structures:

    ::

        data_root
        ├── ref
        │   ├── 0001.png
        │   ├── 0002.png
        ├── gt
        │   ├── 0001.png
        │   ├── 0002.png
        ├── lq
        │   ├── 0001_x4.png
        │   ├── 0002_x4.png

    then, you need to set:

    .. code-block:: python

        ref_folder = 'data_root/ref'
        gt_folder = 'data_root/gt'
        lq_folder = 'data_root/lq'
        filename_tmpl_gt='{}'
        filename_tmpl_lq='{}_x4'

    Args:
        pipeline (List[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        ref_folder (str | :obj:`Path`): Path to a ref folder.
        gt_folder (str | :obj:`Path` | None): Path to a gt folder.
            Default: None.
        lq_folder (str | :obj:`Path` | None): Path to a gt folder.
            Default: None.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        filename_tmpl_gt (str): Template for gt filename. Note that the
            template excludes the file extension. Default: '{}'.
        filename_tmpl_lq (str): Template for lq filename. Note that the
            template excludes the file extension. Default: '{}'.
    """

    def __init__(self,
                 pipeline,
                 scale,
                 ref_folder,
                 gt_folder=None,
                 lq_folder=None,
                 test_mode=False,
                 filename_tmpl_gt='{}',
                 filename_tmpl_lq='{}'):
        super().__init__(pipeline, scale, test_mode)
        assert gt_folder or lq_folder, 'At least one of gt_folder and' \
            'lq_folder cannot be None.'
        self.scale = scale
        self.ref_folder = str(ref_folder)
        self.gt_folder = str(gt_folder) if gt_folder else None
        self.lq_folder = str(lq_folder) if lq_folder else None
        self.filename_tmpl_gt = filename_tmpl_gt
        self.filename_tmpl_lq = filename_tmpl_lq
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for SR dataset.

        It loads the ref, LQ and GT image path from folders.

        Returns:
            list[dict]: A list of dicts for paired paths of ref, LQ and GT.
        """
        data_infos = []
        ref_paths = self.scan_folder(self.ref_folder)
        if self.gt_folder is not None:
            gt_paths = self.scan_folder(self.gt_folder)
            assert len(ref_paths) == len(gt_paths), (
                f'ref and gt datasets have different number of images: '
                f'{len(ref_paths)}, {len(gt_paths)}.')
        if self.lq_folder is not None:
            lq_paths = self.scan_folder(self.lq_folder)
            assert len(ref_paths) == len(lq_paths), (
                f'ref and lq datasets have different number of images: '
                f'{len(ref_paths)}, {len(lq_paths)}.')
        for ref_path in ref_paths:
            basename, ext = osp.splitext(osp.basename(ref_path))
            data_dict = dict(ref_path=ref_path)
            if self.gt_folder is not None:
                gt_path = osp.join(self.gt_folder,
                                   (f'{self.filename_tmpl_gt.format(basename)}'
                                    f'{ext}'))
                assert gt_path in gt_paths, \
                    f'{gt_path} is not in gt_paths.'
                data_dict['gt_path'] = gt_path
            if self.lq_folder is not None:
                lq_path = osp.join(self.lq_folder,
                                   (f'{self.filename_tmpl_lq.format(basename)}'
                                    f'{ext}'))
                assert lq_path in lq_paths, \
                    f'{lq_path} is not in lq_paths.'
                data_dict['lq_path'] = lq_path
            data_infos.append(data_dict)
        return data_infos
