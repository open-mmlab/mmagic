# Copyright (c) OpenMMLab. All rights reserved.
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.dataset import Compose

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
        img_out_dir (str): Output directory of images. Defaults to ''.
        pred_out_file: File to save the inference results. If left as empty, no
            file will be saved.
        print_result (bool): Whether to print the result.
            Defaults to False.
    """

    func_kwargs = dict(preprocess=[], forward=[], visualize=[], postprocess=[])
    func_order = dict(preprocess=0, forward=1, visualize=2, postprocess=3)

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
        self._init_model(cfg, ckpt, device)
        self._init_pipeline(cfg)
        self._init_visualizer(cfg)
        self.base_params = self._dispatch_kwargs(**kwargs)

    def _init_model(self, cfg: Union[ConfigType, str], ckpt: Optional[str],
                    device: str) -> None:
        """Initialize the model with the given config and checkpoint on the
        specific device."""
        model = MODELS.build(cfg.model)
        if ckpt is not None:
            ckpt = load_checkpoint(model, ckpt, map_location='cpu')
        model.cfg = cfg.model
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

    def _dispatch_kwargs(self, **kwargs) -> Tuple[Dict, Dict, Dict, Dict]:
        """Dispatch kwargs to preprocess(), forward(), visualize() and
        postprocess() according to the actual demands."""
        results = [{}, {}, {}, {}]
        dispatched_kwargs = set()

        # Dispatch kwargs according to self.func_kwargs
        for func_name, func_kwargs in self.func_kwargs.items():
            for func_kwarg in func_kwargs:
                if func_kwarg in kwargs:
                    dispatched_kwargs.add(func_kwarg)
                    results[self.func_order[func_name]][func_kwarg] = kwargs[
                        func_kwarg]

        # Find if there is any kwargs that are not dispatched
        for kwarg in kwargs:
            if kwarg not in dispatched_kwargs:
                raise ValueError(f'Unknown kwarg: {kwarg}')

        return results

    def preprocess(self, inputs: InputsType) -> List[Dict]:
        """Process the inputs into a model-feedable format."""
        raise NotImplementedError

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
                  img_out_dir: str = '') -> List[np.ndarray]:
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
            img_out_dir (str): Output directory of images. Defaults to ''.
        """
        raise NotImplementedError

    def postprocess(
        self,
        preds: PredType,
        imgs: Optional[List[np.ndarray]] = None,
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

        Returns:
            TODO
        """
        raise NotImplementedError
