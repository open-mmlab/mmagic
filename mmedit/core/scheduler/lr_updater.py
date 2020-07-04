from mmcv.runner import HOOKS, LrUpdaterHook


@HOOKS.register_module()
class LinearLrUpdaterHook(LrUpdaterHook):
    """Linear learning rate scheduler for image generation.

    In the beginning, the learning rate is 'base_lr' defined in mmcv.
    We give a target learning rate 'target_lr' and a start point 'start'
    (iteration / epoch). Before 'start', we fix learning rate as 'base_lr';
    After 'start', we linearly update learning rate to 'target_lr'.

    Args:
        target_lr (float): The target learning rate. Default: 0.
        start (int): The start point (iteration / epoch, specified by args
            'by_epoch' in its parent class in mmcv) to update learning rate.
            Default: 0.
        interval (int): The interval to update the learning rate. Default: 1.
    """

    def __init__(self, target_lr=0, start=0, interval=1, **kwargs):
        super(LinearLrUpdaterHook, self).__init__(**kwargs)
        self.target_lr = target_lr
        self.start = start
        self.interval = interval

    def get_lr(self, runner, base_lr):
        """Calculates the learning rate.

        Args:
            runner (object): The passed runner.
            base_lr (float): Base learning rate.

        Returns:
            float: Current learning rate.
        """
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        assert max_progress >= self.start
        if max_progress == self.start:
            return base_lr
        else:
            # Before 'start', fix lr; After 'start', linearly update lr.
            factor = (max(0, progress - self.start) // self.interval) / (
                (max_progress - self.start) // self.interval)
            return base_lr + (self.target_lr - base_lr) * factor
