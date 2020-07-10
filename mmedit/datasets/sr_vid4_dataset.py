from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRVid4Dataset(BaseSRDataset):
    """Vid4 dataset for video super resolution.

    The dataset loads several LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a dict containing paired data and other information.

    It reads Vid4 keys from the txt file.
    Each line contains:

        1. folder name;
        2. number of frames in this clip (in the same folder);
        3. image shape, seperated by a white space.

    Examples:

    ::

        calendar 40 (320,480,3)
        city 34 (320,480,3)

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        num_input_frames (int): Window size for input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{:08d}'.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 ann_file,
                 num_input_frames,
                 pipeline,
                 scale,
                 filename_tmpl='{:08d}',
                 test_mode=False):
        super(SRVid4Dataset, self).__init__(pipeline, scale, test_mode)
        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames}.')
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)
        self.num_input_frames = num_input_frames
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for Vid4 dataset.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.strip().split(' ')
                for i in range(int(frame_num)):
                    data_infos.append(
                        dict(
                            lq_path=self.lq_folder,
                            gt_path=self.gt_folder,
                            key=f'{folder}/{self.filename_tmpl.format(i)}',
                            num_input_frames=self.num_input_frames,
                            max_frame_num=int(frame_num)))
        return data_infos
