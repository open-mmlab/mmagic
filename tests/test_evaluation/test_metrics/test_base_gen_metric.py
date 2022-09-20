# Copyright (c) OpenMMLab. All rights reserved.
# TODO:
from mmedit.evaluation.metrics.base_gen_metric import GenMetric


class DemoMetric(GenMetric):

    def process(self, data_batch, data_samples):
        return

    def compute_metrics(self, results):
        return dict(score=1)


def test_GenMetric():
    metric = DemoMetric(10, 10)
    assert metric.real_nums_per_device == 10

    metric.real_results = []
