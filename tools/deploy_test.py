import argparse

import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel

from mmedit.apis import single_gpu_test
from mmedit.core.export import ONNXRuntimeEditing
from mmedit.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('model', help='input model file')
    parser.add_argument('--out', help='output result pickle file')
    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **loader_cfg)

    # build the model
    model = ONNXRuntimeEditing(args.model, cfg=cfg, device_id=0)

    args.save_image = args.save_path is not None
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(
        model,
        data_loader,
        save_path=args.save_path,
        save_image=args.save_image)

    print()
    # print metrics
    stats = dataset.evaluate(outputs)
    for stat in stats:
        print('Eval-{}: {}'.format(stat, stats[stat]))

    # save result pickle
    if args.out:
        print('writing results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)


if __name__ == '__main__':
    main()
