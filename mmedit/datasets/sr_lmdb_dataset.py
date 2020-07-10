import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRLmdbDataset(BaseSRDataset):
    """General paired image lmdb dataset for image restoration.

    The dataset loads lq (Low Quality) and gt (Ground-Truth) image pairs,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "lmdb mode". In order to speed up IO, you are recommended to
    use lmdb. First, you need to make lmdb files. Suppose the lmdb files
    are path_to_lq/lq.lmdb and path_to_gt/gt.lmdb, then you can just set:

    .. code-block:: python

        lq_folder = path_to_lq/lq.lmdb
        gt_folder = path_to_gt/gt.lmdb

    Contents of lmdb. Taking the lq.lmdb for example, the file structure is:

    ::

        lq.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records

        1. image name (with extension);
        2. image shape;
        3. compression level, separated by a white space.

    For example, the meta information of the lq.lmdb is:
    `baboon.png (120,125,3) 1`, which means:
    1) image name (with extension): baboon.png; 2) image shape: (120,125,3);
    and 3) compression level: 1

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq lmdb file.
        gt_folder (str | :obj:`Path`): Path to a gt lmdb file.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self, lq_folder, gt_folder, pipeline, scale, test_mode=False):
        super(SRLmdbDataset, self).__init__(pipeline, scale, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.scale = scale

        if not (self.gt_folder.endswith('.lmdb')
                and self.lq_folder.endswith('.lmdb')):
            raise ValueError(
                f'gt folder and lq folder should both in lmdb format. '
                f'But received gt: {self.gt_folder}; lq: {self.lq_folder}')

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for SR dataset.

        It loads the LQ and GT image path from the ``meta_info.txt`` in the
        LMDB files.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        data_infos = []
        # read keys from meta_info.txt in the gt folder
        # lq and gt keys are the same, ensured by the creation process
        # lq_path and gt_path are replaced by lmdb keys in lmdb mode.
        with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
            for line in fin:
                key = line.split(' ')[0].split('.')[0]
                data_infos.append(dict(lq_path=key, gt_path=key))

        return data_infos
