import glob
import os.path as osp
import shutil
from argparse import ArgumentParser
from copy import deepcopy

from mmengine import Config
from tqdm import tqdm


def update_intervals(config, args):
    if args.iters is None:
        return config

    # 1. change max-iters and val-interval
    if 'train_cfg' in config and config['train_cfg']:
        config['train_cfg'] = dict(
            type='IterBasedTrainLoop',
            max_iters=args.iters,
            val_interval=args.iters // 5)

    # 2. change logging interval
    if 'default_hooks' in config and config['default_hooks']:
        config['default_hooks']['logger'] = dict(
            type='LoggerHook', interval=args.iters // 10)
        config['default_hooks']['checkpoint'] = dict(
            type='CheckpointHook', interval=args.iters // 15)

    return config


def convert_data_config(data_cfg):
    ceph_dataroot_prefix_temp = 'openmmlab:s3://openmmlab/datasets/{}/'
    local_dataroot_prefix = ['data', './data']
    # val_dataloader may None
    if data_cfg is None:
        return

    data_cfg_updated = deepcopy(data_cfg)
    dataset: dict = data_cfg['dataset']

    dataset_type: str = dataset['type']
    if dataset_type in ['ImageNet', 'CIFAR10']:
        repo_name = 'classification'
    else:
        repo_name = 'editing'
    ceph_dataroot_prefix = ceph_dataroot_prefix_temp.format(repo_name)

    if 'data_root' in dataset:
        data_root: str = dataset['data_root']

        for dataroot_prefix in local_dataroot_prefix:
            if data_root.startswith(dataroot_prefix):
                # avoid cvt './data/imagenet' ->
                # openmmlab:s3://openmmlab/datasets/classification//imagenet
                if data_root.startswith(dataroot_prefix + '/'):
                    dataroot_prefix = dataroot_prefix + '/'
                data_root = data_root.replace(dataroot_prefix,
                                              ceph_dataroot_prefix)
                # add '/' at the end
                if not data_root.endswith('/'):
                    data_root = data_root + '/'
                dataset['data_root'] = data_root

    elif 'data_roots' in dataset:
        # specific for pggan dataset, which need a dict of data_roots
        data_roots: dict = dataset['data_roots']
        for k, data_root in data_roots.items():
            for dataroot_prefix in local_dataroot_prefix:
                if data_root.startswith(dataroot_prefix):
                    # avoid cvt './data/imagenet' ->
                    # openmmlab:s3://openmmlab/datasets/classification//imagenet
                    if data_root.startswith(dataroot_prefix + '/'):
                        dataroot_prefix = dataroot_prefix + '/'
                    data_root = data_root.replace(dataroot_prefix,
                                                  ceph_dataroot_prefix)
                    # add '/' at the end
                    if not data_root.endswith('/'):
                        data_root = data_root + '/'
                    data_roots[k] = data_root
        dataset['data_roots'] = data_roots

    else:
        raise KeyError

    if hasattr(dataset, 'pipeline'):
        pipelines = dataset['pipeline']
        for pipeline in pipelines:
            type_ = pipeline['type']
            # only change mmcv(mmcls)'s loading config
            if type_ == 'mmcls.LoadImageFromFile':
                pipeline['file_client_args'] = dict(backend='petrel')
            elif type_ == 'LoadMask':
                if 'mask_list_file' in pipeline['mask_config']:
                    local_mask_path = pipeline['mask_config']['mask_list_file']
                    for dataroot_prefix in local_dataroot_prefix:
                        if local_mask_path.startswith(dataroot_prefix + '/'):
                            dataroot_prefix = dataroot_prefix + '/'
                        local_mask_path = local_mask_path.replace(
                            dataroot_prefix, ceph_dataroot_prefix)
                    pipeline['mask_config']['mask_list_file'] = local_mask_path
                    pipeline['mask_config']['prefix'] = osp.dirname(
                        local_mask_path)
                    pipeline['mask_config']['io_backend'] = 'petrel'
                    pipeline['mask_config']['file_client_kwargs'] = dict(
                        backend='petrel')
            elif type_ == 'RandomLoadResizeBg':
                bg_dir_path = pipeline['bg_dir']
                for dataroot_prefix in local_dataroot_prefix:
                    if bg_dir_path.startswith(dataroot_prefix + '/'):
                        dataroot_prefix = dataroot_prefix + '/'
                    bg_dir_path = bg_dir_path.replace(dataroot_prefix,
                                                      ceph_dataroot_prefix)
                    bg_dir_path = bg_dir_path.replace(repo_name, 'detection')
                pipeline['bg_dir'] = bg_dir_path
            elif type_ == 'CompositeFg':
                fg_dir_path = pipeline['fg_dirs']
                for i, fg in enumerate(fg_dir_path):
                    for dataroot_prefix in local_dataroot_prefix:
                        if fg.startswith(dataroot_prefix + '/'):
                            dataroot_prefix = dataroot_prefix + '/'
                        fg = fg.replace(dataroot_prefix, ceph_dataroot_prefix)
                        pipeline['fg_dirs'][i] = fg

                alpha_dir_path = pipeline['alpha_dirs']
                for i, alpha_dir in enumerate(alpha_dir_path):
                    for dataroot_prefix in local_dataroot_prefix:
                        if alpha_dir.startswith(dataroot_prefix + '/'):
                            dataroot_prefix = dataroot_prefix + '/'
                        alpha_dir = alpha_dir.replace(dataroot_prefix,
                                                      ceph_dataroot_prefix)
                        pipeline['alpha_dirs'][i] = alpha_dir

    data_cfg_updated['dataset'] = dataset
    return data_cfg_updated


def update_ceph_config(filename, args, dry_run=False):
    if filename.startswith(osp.join(args.target_dir, '_base_')):
        # Skip base configs
        return None

    if args.ceph_path is not None:
        if args.ceph_path.endswith('/'):
            args.ceph_path = args.ceph_path[:-1]
        work_dir = f'{args.ceph_path}/{args.work_dir_prefix}'
        save_dir = f'{args.ceph_path}/{args.save_dir_prefix}'
        if not work_dir.endswith('/'):
            work_dir = work_dir + '/'
        if not save_dir.endswith('/'):
            save_dir = save_dir + '/'
    else:
        # disable save local results to ceph
        work_dir = args.work_dir_prefix
        save_dir = args.save_dir_prefix

    try:
        config = Config.fromfile(filename)

        # 0. update intervals
        config = update_intervals(config, args)

        # 1. change dataloader
        dataloader_prefix = [
            f'{p}_dataloader' for p in ['train', 'val', 'test']
        ]

        for prefix in dataloader_prefix:
            if not hasattr(config, prefix):
                continue
            data_cfg = config[prefix]
            if not isinstance(data_cfg, list):
                data_cfg_list = [data_cfg]
                data_cfg_is_list = False
            else:
                data_cfg_list = data_cfg
                data_cfg_is_list = True

            data_cfg_updated_list = [
                convert_data_config(cfg) for cfg in data_cfg_list
            ]
            if data_cfg_is_list:
                config[prefix] = data_cfg_updated_list
            else:
                config[prefix] = data_cfg_updated_list[0]

        # 2. change visualizer
        if hasattr(config, 'vis_backends'):
            # TODO: support upload to ceph
            # for vis_cfg in config['vis_backends']:
            #     if vis_cfg['type'] == 'VisBackend':
            #         vis_cfg['ceph_path'] = work_dir

            # add pavi config
            if args.add_pavi:
                _, project, name = filename.split('/')
                name = name[:-2]
                # check if pavi config is inheritance from _base_
                find_inherit = False
                for vis_cfg in config['vis_backends']:
                    if vis_cfg['type'] == 'PaviVisBackend':
                        vis_cfg['exp_name'] = name
                        vis_cfg['project'] = project
                        find_inherit = True
                        break

                if not find_inherit:
                    pavi_cfg = dict(
                        type='PaviVisBackend', exp_name=name, project=project)
                    config['vis_backends'].append(pavi_cfg)

            # add wandb config
            if args.add_wandb:
                _, project, name = filename.split('/')
                name = name[:-2]
                # check if wandb config is inheritance from _base_
                find_inherit = False
                for vis_cfg in config['vis_backends']:
                    if vis_cfg['type'] == 'WandbVisBackend':
                        vis_cfg['name'] = name  # name of config
                        vis_cfg['project'] = project  # name of model
                        find_inherit = True
                        break

                if not find_inherit:
                    pavi_cfg = dict(
                        type='WandbVisBackend',
                        init_kwargs=dict(name=name, project=project))
                    config['vis_backends'].append(pavi_cfg)

            # add tensorboard config
            if args.add_tensorboard:
                find_inherit = False
                for vis_cfg in config['vis_backends']:
                    if vis_cfg['type'] == 'TensorboardVisBackend':
                        find_inherit = True
                        break

                if not find_inherit:
                    tensorboard_cfg = dict(type='TensorboardVisBackend')
                    config['vis_backends'].append(tensorboard_cfg)

            config['visualizer']['vis_backends'] = config['vis_backends']

        # 3. change logger hook and checkpoint hook
        if hasattr(config, 'default_hooks'):
            # file_client_args = dict(backend='petrel')

            for name, hooks in config['default_hooks'].items():
                if name == 'logger':
                    hooks['out_dir'] = save_dir
                    # hooks['file_client_args'] = file_client_args
                elif name == 'checkpoint':
                    hooks['out_dir'] = save_dir
                    # hooks['file_client_args'] = file_client_args

        # 4. save
        config.dump(config.filename)
        return True

    except Exception as e:  # noqa
        print(e)
        return False


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--ceph-path', type=str, default=None)
    parser.add_argument('--gpus-per-job', type=int, default=None)
    parser.add_argument(
        '--save-dir-prefix',
        type=str,
        default='work_dirs',
        help='Default prefix of the work dirs in the bucket')
    parser.add_argument(
        '--work-dir-prefix',
        type=str,
        default='work_dirs',
        help='Default prefix of the work dirs in the bucket')
    parser.add_argument(
        '--target-dir', type=str, default='configs_ceph', help='configs path')
    parser.add_argument(
        '--iters', type=int, default=None, help='set intervals')
    parser.add_argument(
        '--test-file', type=str, default=None, help='Dry-run on a test file.')
    parser.add_argument(
        '--add-pavi', action='store_true', help='Add pavi config or not.')
    parser.add_argument(
        '--add-wandb', action='store_true', help='Add wandb config or not.')
    parser.add_argument(
        '--add-tensorboard',
        action='store_true',
        help='Add Tensorboard config or not.')

    args = parser.parse_args()

    if args.test_file is None:

        print('Copying config files to "config_ceph" ...')
        shutil.copytree('configs', args.target_dir, dirs_exist_ok=True)

        print('Updating ceph configuration ...')
        files = glob.glob(
            osp.join(args.target_dir, '**', '*.py'), recursive=True)
        pbars = tqdm(files)
        res = []
        for f in pbars:
            pbars.set_description(f'Processing {f}')
            res.append(update_ceph_config(f, args))

        count_skip = res.count(None)
        count_done = res.count(True)
        count_fail = res.count(False)
        fail_list = [fn for status, fn in zip(res, files) if status is False]
        skip_list = [fn for status, fn in zip(res, files) if status is None]

        print(f'Successfully update {count_done} configs.')
        print(f'Skip {count_skip} configs.')
        if count_skip > 0:
            print(skip_list)
        print(f'Fail {count_fail} configs.')
        if count_fail > 0:
            print(fail_list)

    else:
        shutil.copy(args.test_file,
                    args.test_file.replace('configs', args.target_dir))
        update_ceph_config(
            args.test_file.replace('configs', args.target_dir),
            args,
            dry_run=True)
