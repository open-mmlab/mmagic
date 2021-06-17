import glob
import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRFolderMultipleGTDataset(BaseSRDataset):
    """General dataset for video super resolution, used for recurrent networks.

    It assumes all video clips under the root directory is used for training
    or test.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        num_input_frames (None | int): The number of frames per iteration.
            If None, the whole clip is extracted. If it is a positive integer,
            a sequence of 'num_input_frames' frames is extracted from the clip.
            Note that non-positive integers are not accepted. Default: None.
        test_mode (bool): Store `True` when building test dataset.
            Default: `True`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 scale,
                 num_input_frames=None,
                 test_mode=True):
        super().__init__(pipeline, scale, test_mode)

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)

        if num_input_frames is not None and num_input_frames <= 0:
            raise ValueError('"num_input_frames" must be None or positive, '
                             f'but got {num_input_frames}.')
        self.num_input_frames = num_input_frames

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for the dataset.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """

        sequences = sorted(glob.glob(osp.join(self.lq_folder, '*')))
        data_infos = []
        for sequence in sequences:
            sequence_length = len(glob.glob(osp.join(sequence, '*.png')))
            if self.num_input_frames is None:
                num_input_frames = sequence_length
            else:
                num_input_frames = self.num_input_frames
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=sequence.replace(f'{self.lq_folder}/', ''),
                    num_input_frames=num_input_frames,
                    sequence_length=sequence_length))

        return data_infos
