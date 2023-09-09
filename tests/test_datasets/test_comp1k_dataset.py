# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
from pathlib import Path

from mmagic.datasets import AdobeComp1kDataset


class TestMattingDatasets:

    # path to a minimal test dataset under 'tests/data'
    DATA_ROOT = Path(__file__).parent.parent / 'data' / 'matting_dataset'

    # cls.pipeline = [
    #     dict(type='LoadImageFromFile', key='alpha', flag='grayscale')
    # ]

    def test_comp1k_dataset(self):
        """Verify AdobeComp1kDataset reads dataset correctly.

        AdobeComp1kDataset should support both new and old annotation formats.
        """

        ann_new = 'ann.json'
        ann_old = 'ann_old.json'

        for ann in ann_new, ann_old:
            # TODO: we may add an actual pipeline later
            ds = AdobeComp1kDataset(ann, data_root=self.DATA_ROOT, pipeline=[])

            assert len(ds) == 2
            assert ds.metainfo == dict(
                dataset_type='matting_dataset', task_name='matting')

            data0 = ds[0]
            data1 = ds[1]
            assert osp.isfile(data0['alpha_path'])
            assert osp.isfile(data1['alpha_path'])

            is_correct_ntpath = data0 == {
                'alpha_path': f'{self.DATA_ROOT}\\alpha/GT05.jpg',
                'trimap_path': f'{self.DATA_ROOT}\\trimap/GT05.jpg',
                'bg_path': f'{self.DATA_ROOT}\\bg/GT26r.jpg',
                'fg_path': f'{self.DATA_ROOT}\\fg/GT05.jpg',
                'merged_path': f'{self.DATA_ROOT}\\merged/GT05.jpg',
                'sample_idx': 0,
            }
            is_correct_posixpath = data0 == {
                'alpha_path': f'{self.DATA_ROOT}/alpha/GT05.jpg',
                'trimap_path': f'{self.DATA_ROOT}/trimap/GT05.jpg',
                'bg_path': f'{self.DATA_ROOT}/bg/GT26r.jpg',
                'fg_path': f'{self.DATA_ROOT}/fg/GT05.jpg',
                'merged_path': f'{self.DATA_ROOT}/merged/GT05.jpg',
                'sample_idx': 0,
            }
            assert is_correct_ntpath or is_correct_posixpath

            is_correct_ntpath = data1 == {
                'alpha_path': f'{self.DATA_ROOT}\\alpha/GT05.jpg',
                'bg_path': f'{self.DATA_ROOT}\\bg/GT26r.jpg',
                'fg_path': f'{self.DATA_ROOT}\\fg/GT05.jpg',
                'sample_idx': 1,
            }
            is_correct_posixpath = data1 == {
                'alpha_path': f'{self.DATA_ROOT}/alpha/GT05.jpg',
                'bg_path': f'{self.DATA_ROOT}/bg/GT26r.jpg',
                'fg_path': f'{self.DATA_ROOT}/fg/GT05.jpg',
                'sample_idx': 1,
            }
            assert is_correct_ntpath or is_correct_posixpath

            # TODO: after mmcv.transform becomes ready in CI
            # assert 'alpha' in first_data
            # assert isinstance(first_data['alpha'], np.ndarray)
            # assert first_data['alpha'].shape == (552, 800)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
