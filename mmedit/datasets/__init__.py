# Copyright (c) OpenMMLab. All rights reserved.
from .base_dataset import BaseDataset
from .base_generation_dataset import BaseGenerationDataset
from .base_matting_dataset import BaseMattingDataset
from .base_sr_dataset import BaseSRDataset
from .base_vfi_dataset import BaseVFIDataset
from .builder import build_dataloader, build_dataset
from .comp1k_dataset import AdobeComp1kDataset
from .dataset_wrappers import RepeatDataset
from .generation_paired_dataset import GenerationPairedDataset
from .generation_unpaired_dataset import GenerationUnpairedDataset
from .img_inpainting_dataset import ImgInpaintingDataset
from .registry import DATASETS, PIPELINES
from .sr_annotation_dataset import SRAnnotationDataset
from .sr_facial_landmark_dataset import SRFacialLandmarkDataset
from .sr_folder_dataset import SRFolderDataset
from .sr_folder_gt_dataset import SRFolderGTDataset
from .sr_folder_multiple_gt_dataset import SRFolderMultipleGTDataset
from .sr_folder_ref_dataset import SRFolderRefDataset
from .sr_folder_video_dataset import SRFolderVideoDataset
from .sr_lmdb_dataset import SRLmdbDataset
from .sr_reds_dataset import SRREDSDataset
from .sr_reds_multiple_gt_dataset import SRREDSMultipleGTDataset
from .sr_test_multiple_gt_dataset import SRTestMultipleGTDataset
from .sr_vid4_dataset import SRVid4Dataset
from .sr_vimeo90k_dataset import SRVimeo90KDataset
from .sr_vimeo90k_multiple_gt_dataset import SRVimeo90KMultipleGTDataset
from .vfi_vimeo90k_7frames_dataset import VFIVimeo90K7FramesDataset
from .vfi_vimeo90k_dataset import VFIVimeo90KDataset

__all__ = [
    'DATASETS', 'PIPELINES', 'build_dataset', 'build_dataloader',
    'BaseDataset', 'BaseMattingDataset', 'ImgInpaintingDataset',
    'AdobeComp1kDataset', 'SRLmdbDataset', 'SRFolderDataset',
    'SRAnnotationDataset', 'BaseSRDataset', 'RepeatDataset', 'SRREDSDataset',
    'SRVimeo90KDataset', 'BaseGenerationDataset', 'GenerationPairedDataset',
    'GenerationUnpairedDataset', 'SRVid4Dataset', 'SRFolderGTDataset',
    'SRREDSMultipleGTDataset', 'SRVimeo90KMultipleGTDataset',
    'SRTestMultipleGTDataset', 'SRFolderRefDataset', 'SRFacialLandmarkDataset',
    'SRFolderMultipleGTDataset', 'SRFolderVideoDataset', 'BaseVFIDataset',
    'VFIVimeo90KDataset', 'VFIVimeo90K7FramesDataset'
]
