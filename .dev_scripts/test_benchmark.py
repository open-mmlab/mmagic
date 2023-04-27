import argparse
import os
import os.path as osp
import pickle
import re
from collections import OrderedDict, defaultdict
from datetime import datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path

from job_watcher import start_from_proc
from metric_mapping import METRICS_MAPPING, filter_metric
from modelindex.load_model_index import load
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from task_mapping import TASK_MAPPING

console = Console()
MMagic_ROOT = Path(__file__).absolute().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test all models' accuracy in model-index.yml")
    parser.add_argument(
        'partition', type=str, help='Cluster partition to use.')
    parser.add_argument('checkpoint_root', help='Checkpoint file root path.')
    parser.add_argument(
        '--job-name', type=str, default=' ', help='Slurm job name prefix')
    parser.add_argument('--port', type=int, default=29666, help='dist port')
    parser.add_argument(
        '--use-ceph-config',
        action='store_true',
        default=False,
        help='Use ceph configs or not.')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/benchmark_test',
        help='the dir to save metric')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--local',
        action='store_true',
        help='run at local instead of cluster.')
    parser.add_argument(
        '--mail', type=str, help='Mail address to watch test status.')
    parser.add_argument(
        '--mail-type',
        nargs='+',
        default=['BEGIN'],
        choices=['NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'],
        help='Mail address to watch test status.')
    parser.add_argument(
        '--quotatype',
        default=None,
        choices=['reserved', 'auto', 'spot'],
        help='Quota type, only available for phoenix-slurm>=0.2')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Summarize benchmark test results.')
    parser.add_argument(
        '--by-task',
        action='store_true',
        help='Summairze benchmark results by task.')
    parser.add_argument('--save', action='store_true', help='Save the summary')

    group_parser = parser.add_mutually_exclusive_group()
    group_parser.add_argument(
        '--P0', action='store_true', help='Whether test model in P0 list')
    group_parser.add_argument(
        '--model-list',
        type=str,
        default='',
        help='Path of algorithm list to load')
    args = parser.parse_args()
    return args


def create_test_job_batch(commands, model_info, args, port, script_name):
    config_http_prefix_blob = ('https://github.com/open-mmlab/mmagic/'
                               'blob/main/')
    config_http_prefix_tree = ('https://github.com/open-mmlab/mmagic/'
                               'tree/main/')
    fname = model_info.name

    config = model_info.config
    if config.startswith('http'):
        config = config.replace(config_http_prefix_blob, './')
        config = config.replace(config_http_prefix_tree, './')
    if args.use_ceph_config:
        config = config.replace('configs', 'configs_ceph')

    config = Path(config)
    assert config.exists(), f'{fname}: {config} not found.'

    http_prefix_short = 'https://download.openmmlab.com/mmediting/'
    model_weight_url = model_info.weights

    if model_weight_url.startswith(http_prefix_short):
        model_name = model_weight_url[len(http_prefix_short):]
    elif model_weight_url == '':
        print(f'{fname} weight is missing')
        return None
    else:
        raise ValueError(f'Unknown url prefix. \'{model_weight_url}\'')

    model_name_split = model_name.split('/')
    if len(model_name_split) == 3:  # 'TASK/METHOD/MODEL.pth'
        # remove task name
        model_name = osp.join(*model_name_split[1:])
    else:
        model_name = osp.join(*model_name_split)

    if 's3://' in args.checkpoint_root:
        from mmengine.fileio import FileClient
        from petrel_client.common.exception import AccessDeniedError
        file_client = FileClient.infer_client(uri=args.checkpoint_root)
        checkpoint = file_client.join_path(args.checkpoint_root, model_name)
        try:
            exists = file_client.exists(checkpoint)
        except AccessDeniedError:
            exists = False
    else:
        checkpoint_root = Path(args.checkpoint_root)
        checkpoint = checkpoint_root / model_name
        exists = checkpoint.exists()
    if not exists:
        print(f'WARNING: {fname}: {checkpoint} not found.')
        return None

    job_name = f'{args.job_name}_{fname}'.strip('_')
    work_dir = Path(args.work_dir) / fname
    work_dir.mkdir(parents=True, exist_ok=True)
    result_file = work_dir / 'result.pkl'

    if args.mail is not None and 'NONE' not in args.mail_type:
        mail_cfg = (f'#SBATCH --mail {args.mail}\n'
                    f'#SBATCH --mail-type {args.mail_type}\n')
    else:
        mail_cfg = ''

    if args.quotatype is not None:
        quota_cfg = f'#SBATCH --quotatype {args.quotatype}\n'
    else:
        quota_cfg = ''

    launcher = 'none' if args.local else 'slurm'
    runner = 'python' if args.local else 'srun python'

    # NOTE: set gpus as 2
    job_script = (f'#!/bin/bash\n'
                  f'#SBATCH --output {work_dir}/job.%j.out\n'
                  f'#SBATCH --partition={args.partition}\n'
                  f'#SBATCH --job-name {job_name}\n'
                  f'#SBATCH --gres=gpu:2\n'
                  f'{mail_cfg}{quota_cfg}'
                  f'#SBATCH --ntasks-per-node=2\n'
                  f'#SBATCH --ntasks=2\n'
                  f'#SBATCH --cpus-per-task=16\n\n'
                  f'export MASTER_PORT={port}\n'
                  f'export CUBLAS_WORKSPACE_CONFIG=:4096:8\n'
                  f'{runner} -u {script_name} {config} {checkpoint} '
                  f'--work-dir={work_dir} '
                  f'--out={result_file} '
                  f'--launcher={launcher}\n')

    with open(work_dir / 'job.sh', 'w') as f:
        f.write(job_script)

    commands.append(f'echo "{config}"')
    commands.append(f'echo "{work_dir}"')
    if args.local:
        commands.append(f'bash {work_dir}/job.sh')
    else:
        commands.append(f'sbatch {work_dir}/job.sh')

    return work_dir / 'job.sh'


def test(args):
    # parse model-index.yml
    model_index_file = MMagic_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    script_name = osp.join('tools', 'test.py')
    port = args.port

    commands = []
    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    # load model list
    if args.P0:
        file_list = osp.join(osp.dirname(__file__), 'p0_test_list.py')
    elif args.model_list:
        file_list = args.model_list
    else:
        file_list = None

    if file_list:
        test_list = SourceFileLoader('model_list',
                                     file_list).load_module().model_list
    else:
        test_list = None

    preview_script = ''
    for model_info in models.values():

        if model_info.results is None:
            continue

        if test_list is not None and model_info.name not in test_list:
            continue

        script_path = create_test_job_batch(commands, model_info, args, port,
                                            script_name)
        preview_script = script_path or preview_script
        port += 1

    command_str = '\n'.join(commands)

    preview = Table()
    preview.add_column(str(preview_script))
    preview.add_column('Shell command preview')
    preview.add_row(
        Syntax.from_path(
            preview_script,
            background_color='default',
            line_numbers=True,
            word_wrap=True),
        Syntax(
            command_str,
            'bash',
            background_color='default',
            line_numbers=True,
            word_wrap=True))
    console.print(preview)

    if args.run:
        proc = os.popen(command_str)
        job_name_list = start_from_proc(args.work_dir, proc)
        history_log = datetime.now().strftime('test-%Y%m%d-%H%M%S') + '.log'
        with open(history_log, 'w') as fp:
            fp.write(args.work_dir + '\n')
            for job in job_name_list:
                fp.write(job + '\n')

        cache_path = osp.expanduser(osp.join('~', '.task_watcher'))
        # print(cache_path)
        os.makedirs(cache_path, exist_ok=True)
        with open(osp.join(cache_path, 'latest.log'), 'w') as fp:
            fp.write(args.work_dir + '\n')
            for job in job_name_list:
                fp.write(job + '\n')
        print(f'Have saved job submission history in {history_log}')
    else:
        console.print('Please set "--run" to start the job')


def show_summary(summary_data,
                 models_map,
                 work_dir,
                 name='test_benchmark_summary',
                 save=False):
    # table = Table(title='Test Benchmark Regression Summary')
    table_title = name.replace('_', ' ')
    table_title = table_title.capitalize()
    table = Table(title=table_title)
    table.add_column('Model')
    md_header = ['Model']

    used_metrics = filter_metric(METRICS_MAPPING, summary_data)
    for metric in used_metrics:
        table.add_column(f'{metric} (expect)')
        table.add_column(f'{metric}')
        md_header.append(f'{metric} (expect)')
        md_header.append(f'{metric}')
    table.add_column('Date')
    md_header.append('Config')

    def set_color(value, expect, tolerance, rule):
        if value > expect + tolerance:
            return 'green' if rule == 'larger' else 'red'
        elif value < expect - tolerance:
            return 'red' if rule == 'larger' else 'green'
        else:
            return 'white'

    md_rows = ['| ' + ' | '.join(md_header) + ' |\n']
    md_rows.append('|:' + ':|:'.join(['---'] * len(md_header)) + ':|\n')

    for model_name, summary in summary_data.items():
        row = [model_name]
        md_row = [model_name]
        for metric_key in used_metrics:
            if metric_key in summary:
                metric = summary[metric_key]
                expect = round(metric['expect'], 2)
                result = round(metric['result'], 2)
                tolerance = metric['tolerance']
                rule = metric['rule']
                color = set_color(result, expect, tolerance, rule)
                row.append(f'{expect:.2f}')
                row.append(f'[{color}]{result:.2f}[/{color}]')
                md_row.append(f'{expect:.4f}')
                md_row.append(f'{result:.4f}')
            else:
                row.extend([''] * 2)
                md_row.extend([''] * 2)
        if 'date' in summary:
            row.append(summary['date'])
            md_row.append(summary['date'])
        else:
            row.append('')
            md_row.append('')
        table.add_row(*row)

        # add config to row
        model_info = models_map[model_name]
        md_row.append(model_info.config)
        md_rows.append('| ' + ' | '.join(md_row) + ' |\n')

    console.print(table)

    if save:
        summary_path = work_dir / f'{name}.md'
        with open(summary_path, 'w') as file:
            file.write('# Test Benchmark Regression Summary\n')
            file.writelines(md_rows)


def summary(args):
    model_index_file = MMagic_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    work_dir = Path(args.work_dir)

    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    summary_data = {}
    task_summary_data = defaultdict(dict)
    for model_name, model_info in models.items():

        if model_info.results is None:
            continue

        # Skip if not found result file.
        result_file = work_dir / model_name / 'result.pkl'
        if not result_file.exists():
            summary_data[model_name] = {}
            continue

        with open(result_file, 'rb') as file:
            results = pickle.load(file)
        date = datetime.fromtimestamp(result_file.lstat().st_mtime)

        expect_metrics = model_info.results[0].metrics

        # extract metrics
        summary = {'date': date.strftime('%Y-%m-%d')}
        for key_yml, value_yml in METRICS_MAPPING.items():
            key_results = value_yml['keys']
            tolerance = value_yml['tolerance']
            rule = value_yml['rule']

            for key_result in key_results:
                if key_yml in expect_metrics and key_result in results:
                    expect_result = float(expect_metrics[key_yml])
                    result = float(results[key_result])
                    summary[key_yml] = dict(
                        expect=expect_result,
                        result=result,
                        tolerance=tolerance,
                        rule=rule)

        summary_data[model_name] = summary

        in_collection = model_info.data['In Collection']
        for task, collection_list in TASK_MAPPING.items():
            if in_collection.upper() in [c.upper() for c in collection_list]:
                task_summary_data[task][model_name] = summary
                break

    if args.by_task:
        for task_name, data in task_summary_data.items():
            show_summary(
                data, models, work_dir, f'{task_name}_summary', save=args.save)

    show_summary(summary_data, models, work_dir, save=args.save)


def main():
    args = parse_args()

    if args.summary:
        summary(args)
    else:
        test(args)


if __name__ == '__main__':
    main()
