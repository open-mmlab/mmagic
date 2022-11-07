# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from mmengine.structures import InstanceData

from mmedit.utils import ConfigType
from .base_inferencer import BaseInferencer

InstanceList = List[InstanceData]
InputType = Union[str, int, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


class BaseMMEditInferencer(BaseInferencer):
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

    func_kwargs = dict(
        preprocess=[],
        forward=[],
        visualize=[
            'show', 'wait_time', 'draw_pred', 'pred_score_thr', 'result_out_dir'
        ],
        postprocess=['print_result', 'pred_out_file', 'get_datasample'])
    func_order = dict(preprocess=0, forward=1, visualize=2, postprocess=3)
    

    def __init__(self,
                 config: Union[ConfigType, str],
                 ckpt: Optional[str],
                 device: Optional[str] = None,
                 **kwargs) -> None:
        self.base_params = self._dispatch_kwargs(**kwargs)
        super().__init__(config=config, ckpt=ckpt, device=device, **kwargs)

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

        return results

    def __call__(self, **kwargs) -> Union[Dict, List[Dict]]:
        """Call the inferencer.

        Args:
            kwargs: Keyword arguments for the inferencer.
        """

        params = self._dispatch_kwargs(**kwargs)
        preprocess_kwargs = self.base_params[0].copy()
        preprocess_kwargs.update(params[0])
        forward_kwargs = self.base_params[1].copy()
        forward_kwargs.update(params[1])
        visualize_kwargs = self.base_params[2].copy()
        visualize_kwargs.update(params[2])
        postprocess_kwargs = self.base_params[3].copy()
        postprocess_kwargs.update(params[3])

        data = self.preprocess(**preprocess_kwargs)
        preds = self.forward(data, **forward_kwargs)
        imgs = self.visualize(preds, data, **visualize_kwargs)
        results = self.postprocess(
            preds, imgs, **postprocess_kwargs)
        return results




