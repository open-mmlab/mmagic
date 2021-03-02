"""This file is for benchmark dataloading process. The command line to run this
file is:
$ python -m cProfile -o program.prof tools/analysis/bench_processing.py
configs/task/method/[config filename]
It use cProfile to record cpu running time and output to program.prof
To visualize cProfile output program.prof, use Snakeviz and run:
$ snakeviz program.prof
"""
import argparse

import mmcv
from mmcv import Config

from mmedit.datasets import build_dataloader, build_dataset
from mmedit.utils import get_root_logger


def main():
    parser = argparse.ArgumentParser(description='Benchmark dataloading')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    # init logger before other steps
    logger = get_root_logger()
    logger.info(f'Config: {cfg.text}')

    dataset = build_dataset(cfg.data.train)
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            dist=False,
            drop_last=cfg.data.get('drop_last', False),
            seed=0) for ds in dataset
    ]
    # Start progress bar after first 5 batches
    prog_bar = mmcv.ProgressBar(
        len(dataset) - 5 * cfg.data.samples_per_gpu, start=False)
    for data_loader in data_loaders:
        for i, data in enumerate(data_loader):
            if i == 5:
                prog_bar.start()
            for _ in data['imgs']:
                if i < 5:
                    continue
                prog_bar.update()


if __name__ == '__main__':
    main()
