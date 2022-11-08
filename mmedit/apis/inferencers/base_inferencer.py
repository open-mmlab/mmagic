# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData

from mmedit.registry import MODELS, VISUALIZERS
from mmedit.utils import ConfigType

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict]]


class BaseInferencer:
    """Base inferencer.

    Args:
        model (str or ConfigType): Model config or the path to it.
        ckpt (str, optional): Path to the checkpoint.
        device (str, optional): Device to run inference. If None, the best
            device will be automatically used.
        show (bool): Whether to display the image in a popup window.
            Defaults to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        draw_pred (bool): Whether to draw predicted bounding boxes.
            Defaults to True.
        pred_score_thr (float): Minimum score of bboxes to draw.
            Defaults to 0.3.
        result_out_dir (str): Output directory of images. Defaults to ''.
        pred_out_file: File to save the inference results. If left as empty, no
            file will be saved.
        print_result (bool): Whether to print the result.
            Defaults to False.
    """

    def __init__(self,
                 config: Union[ConfigType, str],
                 ckpt: Optional[str],
                 device: Optional[str] = None,
                 **kwargs) -> None:
        # Load config to cfg
        if isinstance(config, str):
            cfg = Config.fromfile(config)
        elif not isinstance(config, ConfigType):
            raise TypeError('config must be a filename or any ConfigType'
                            f'object, but got {type(cfg)}')
        self.cfg = cfg
        if cfg.model.get('pretrained'):
            cfg.model.pretrained = None

        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self._init_model(cfg, ckpt, device)
        self._init_visualizer(cfg)

        # A global counter tracking the number of images processed, for
        # naming of the output images
        self.num_visualized_imgs = 0

    def _init_model(self, cfg: Union[ConfigType, str], ckpt: Optional[str],
                    device: str) -> None:
        """Initialize the model with the given config and checkpoint on the
        specific device."""
        model = MODELS.build(cfg.model)
        if ckpt is not None:
            ckpt = load_checkpoint(model, ckpt, map_location='cpu')
        model.cfg = cfg
        model.to(device)
        model.eval()
        self.model = model

    def _init_pipeline(self, cfg: ConfigType) -> None:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        self.file_pipeline = Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1

    def _init_visualizer(self, cfg: ConfigType) -> None:
        """Initialize visualizers."""
        # TODO: We don't export images via backends since the interface
        # of the visualizer will have to be refactored.
        self.visualizer = None
        if 'visualizer' in cfg:
            ts = str(datetime.timestamp(datetime.now()))
            cfg.visualizer['name'] = f'inferencer{ts}'
            self.visualizer = VISUALIZERS.build(cfg.visualizer)

    def preprocess(self, inputs: InputsType) -> Dict:
        """Process the inputs into a model-feedable format."""
        self._init_pipeline(self.cfg)

        results = []
        for single_input in inputs:
            if isinstance(single_input, str):
                if osp.isdir(single_input):
                    raise ValueError('Feeding a directory is not supported')
                else:
                    data_ = dict(img_path=single_input)
                    results.append(self.file_pipeline(data_))
            elif isinstance(single_input, np.ndarray):
                data_ = dict(img=single_input)
                results.append(self.ndarray_pipeline(data_))
            else:
                raise ValueError(
                    f'Unsupported input type: {type(single_input)}')

        return self._collate(results)

    def _collate(self, results: List[Dict]) -> Dict:
        """Collate the results from different images."""
        results = {key: [d[key] for d in results] for key in results[0]}
        return results

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model."""
        with torch.no_grad():
            return self.model.test_step(inputs)

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  result_out_dir: str = '') -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            result_out_dir (str): Output directory of images. Defaults to ''.
        """
        if self.visualizer is None or not show and result_out_dir == '':
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img = mmcv.imread(single_input)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            out_file = osp.join(result_out_dir, img_name) if \
                result_out_dir != '' else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
            )
            results.append(img)
            self.num_visualized_imgs += 1

        return results

    def postprocess(
        self,
        preds: PredType,
        imgs: Optional[List[np.ndarray]] = None,
        is_batch: bool = False,
        print_result: bool = False,
        pred_out_file: str = '',
        get_datasample: bool = False,
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            imgs (Optional[np.ndarray]): Visualized predictions.
            is_batch (bool): Whether the inputs are in a batch.
                Defaults to False.
            print_result (bool): Whether to print the result.
                Defaults to False.
            pred_out_file (str): Output file name to store predictions
                without images. Supported file formats are “json”, “yaml/yml”
                and “pickle/pkl”. Defaults to ''.
            get_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.

        Returns:
            TODO
        """

        results = preds
        if not get_datasample:
            results = []
            for pred in preds:
                result = self._pred2dict(pred)
                results.append(result)
        if not is_batch:
            results = results[0]
        if print_result:
            print(results)
        # Add img to the results after printing
        if pred_out_file != '':
            mmcv.dump(results, pred_out_file)
        if imgs is None:
            return results
        return results, imgs

    def _pred2dict(self, data_sample: torch.Tensor) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary. It's better to contain only basic data elements such as
        strings and numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (torch.Tensor): The data sample to be converted.

        Returns:
            dict: The output dictionary.
        """
        result = {}
        result['infer_res'] = data_sample
        return result
