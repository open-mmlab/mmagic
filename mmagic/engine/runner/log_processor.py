# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.runner import LogProcessor as BaseLogProcessor

from mmagic.registry import LOG_PROCESSORS


@LOG_PROCESSORS.register_module()  # type: ignore
class LogProcessor(BaseLogProcessor):
    """LogProcessor inherits from :class:`mmengine.runner.LogProcessor` and
    overwrites :meth:`self.get_log_after_iter`.

    This log processor should be used along with
    :class:`mmagic.engine.runner.MultiValLoop` and
    :class:`mmagic.engine.runner.MultiTestLoop`.
    """

    def _get_dataloader_size(self, runner, mode) -> int:
        """Get dataloader size of current loop. In `MultiValLoop` and
        `MultiTestLoop`, we use `total_length` instead of `len(dataloader)` to
        denote the total number of iterations.

        Args:
            runner (Runner): The runner of the training/validation/testing
            mode (str): Current mode of runner.

        Returns:
            int: The dataloader size of current loop.
        """
        if hasattr(self._get_cur_loop(runner, mode), 'total_length'):
            return self._get_cur_loop(runner, mode).total_length
        else:
            return super()._get_dataloader_size(runner, mode)
