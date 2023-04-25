import argparse
import os
import os.path as osp
import pickle
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from job_watcher import start_from_proc
from modelindex.load_model_index import load
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from tqdm import tqdm
from utils import filter_jobs, parse_job_list_from_file

console = Console()
MMAGIC_ROOT = Path(__file__).absolute().parents[1]

# key-in-metafile: key-in-results.pkl
METRICS_MAP = {
    'SWD': {
        'keys': ['SWD/avg'],
        'tolerance': 0.1,
        'rule': 'less'
    },
    'MS-SSIM': {
        'keys': ['MS-SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'FID': {
        'keys': ['FID-Full-50k/fid'],
        'tolerance': 0.1,
        'rule': 'less'
    },
    'FID50k': {
        'keys': ['FID-Full-50k/fid'],
        'tolerance': 0.1,
        'rule': 'less'
    },
    'IS': {
        'keys': ['IS-50k/is'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'IS50k': {
        'keys': ['IS-50k/is'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Precision50k': {
        'keys': ['PR-50K/precision'],
        'tolerance': 0.1,
        'rule': 'large'
    },
    'Recall50k': {
        'keys': ['PR-50K/recall'],
        'tolerance': 0.1,
        'rule': 'large'
    },
    'Precision10k': {
        'keys': ['PR-10K/precision'],
        'tolerance': 0.1,
        'rule': 'large'
    },
    'Recall10k': {
        'keys': ['PR-10K/recall'],
        'tolerance': 0.1,
        'rule': 'large'
    },
    # 'PPL': {},  # TODO: no ppl in metafiles?
    'EQ-R': {
        'keys': ['EQ/eqr'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'EQ-T': {
        'keys': ['EQ/eqt_int'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train models' accuracy in model-index.yml")
    parser.add_argument(
        'partition', type=str, help='Cluster partition to use.')
    parser.add_argument('--skip', type=str, default=None)
    parser.add_argument('--skip-list', default=None)
    parser.add_argument('--rerun', type=str, default=None)
    parser.add_argument(
        '--rerun-fail', action='store_true', help='only rerun failed tasks')
    parser.add_argument(
        '--rerun-cancel', action='store_true', help='only rerun cancel tasks')
    parser.add_argument('--rerun-list', default=None)
    parser.add_argument('--gpus-per-job', type=int, default=None)
    parser.add_argument('--cpus-per-job', type=int, default=16)
    parser.add_argument(
        '--amp', action='store_true', help='Whether to use amp.')
    parser.add_argument(
        '--resume', action='store_true', help='Whether to resume checkpoint.')
    parser.add_argument(
        '--job-name', type=str, default=' ', help='Slurm job name prefix')
    parser.add_argument('--port', type=int, default=29666, help='dist port')
    parser.add_argument(
        '--config-dir',
        type=str,
        default='configs_ceph',
        help='Use ceph configs or not.')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/benchmark_train',
        help='the dir to save metric')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Whether set `deterministic` during training.')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--local',
        action='store_true',
        help='run at local instead of cluster.')
    parser.add_argument(
        '--mail', type=str, help='Mail address to watch train status.')
    parser.add_argument(
        '--mail-type',
        nargs='+',
        default=['BEGIN'],
        choices=['NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'],
        help='Mail address to watch train status.')
    parser.add_argument(
        '--quotatype',
        default=None,
        choices=['reserved', 'auto', 'spot'],
        help='Quota type, only available for phoenix-slurm>=0.2')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Summarize benchmark train results.')
    parser.add_argument('--save', action='store_true', help='Save the summary')

    args = parser.parse_args()

    if args.skip is not None:
        with open(args.skip, 'r') as fp:
            skip_list = fp.readlines()
            skip_list = [j.split('\n')[0] for j in skip_list]
            args.skip_list = skip_list
            print('skip_list: ', args.skip_list)
    elif args.rerun is not None:
        job_id_list_full, job_name_list_full = parse_job_list_from_file(
            args.rerun)
        filter_target = []

        if args.rerun_fail:
            filter_target += ['FAILED']
        if args.rerun_cancel:
            filter_target += ['CANCELLED']

        _, job_name_list = filter_jobs(
            job_id_list_full,
            job_name_list_full,
            filter_target,
            show_table=True,
            table_name='Rerun List')
        args.rerun_list = job_name_list

    return args


def create_train_job_batch(commands, model_info, args, port, script_name):
    config_http_prefix_blob = ('https://github.com/open-mmlab/mmagic/'
                               'blob/main/')
    config_http_prefix_tree = ('https://github.com/open-mmlab/mmagic/'
                               'tree/main/')
    fname = model_info.name

    config = model_info.config
    if config.startswith('http'):
        config = config.replace(config_http_prefix_blob, './')
        config = config.replace(config_http_prefix_tree, './')

    config = config.replace('configs', args.config_dir)

    config = Path(config)
    assert config.exists(), f'{fname}: {config} not found.'

    try:
        n_gpus = int(model_info.metadata.data['GPUs'].split()[0])
    except Exception:
        if 'official' in model_info.config:
            return None
        else:
            pattern = r'\d+xb\d+'
            parse_res = re.search(pattern, config.name)
            if not parse_res:
                # defaults to use 1 gpu
                n_gpus = 1
            else:
                n_gpus = int(parse_res.group().split('x')[0])

    if args.gpus_per_job is not None:
        n_gpus = min(args.gpus_per_job, n_gpus)

    job_name = f'{args.job_name}_{fname}'
    if (args.skip_list is not None) and model_info.name in args.skip_list:
        return None
    if (args.rerun_list is not None) and (model_info.name
                                          not in args.rerun_list):
        return None

    work_dir = Path(args.work_dir) / fname
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.mail is not None and 'NONE' not in args.mail_type:
        mail_cfg = (f'#SBATCH --mail {args.mail}\n'
                    f'#SBATCH --mail-type {args.mail_type}\n')
    else:
        mail_cfg = ''

    if args.quotatype is not None:
        quota_cfg = f'#SBATCH --quotatype {args.quotatype}\n'
    else:
        quota_cfg = ''

    launcher = 'none' if args.local or n_gpus == 0 else 'slurm'
    runner = 'python' if args.local else 'srun python'

    job_script = (f'#!/bin/bash\n'
                  f'#SBATCH --output {work_dir}/job.%j.out\n'
                  f'#SBATCH --partition={args.partition}\n'
                  f'#SBATCH --job-name {job_name}\n'
                  f'{mail_cfg}{quota_cfg}')

    if n_gpus > 0:
        job_script += (f'#SBATCH --gres=gpu:{n_gpus}\n'
                       f'#SBATCH --ntasks-per-node={min(n_gpus, 8)}\n'
                       f'#SBATCH --ntasks={n_gpus}\n'
                       f'#SBATCH --cpus-per-task={args.cpus_per_job}\n\n')
    else:
        job_script += '\n\n' + 'export CUDA_VISIBLE_DEVICES=-1\n'

    if args.deterministic:
        job_script += 'export CUBLAS_WORKSPACE_CONFIG=:4096:8\n'

    job_script += (f'export MASTER_PORT={port}\n'
                   f'{runner} -u {script_name} {config} '
                   f'--work-dir={work_dir} '
                   f'--launcher={launcher}')

    if args.resume:
        job_script += '  --resume '

    if args.amp:
        job_script += ' --amp  '

    if args.deterministic:
        job_script += ' --cfg-options randomness.deterministic=True'

    job_script += '\n'

    with open(work_dir / 'job.sh', 'w') as f:
        f.write(job_script)

    commands.append(f'echo "{config}"')
    commands.append(f'echo "{work_dir}"')
    if args.local:
        commands.append(f'bash {work_dir}/job.sh')
    else:
        commands.append(f'sbatch {work_dir}/job.sh')

    return work_dir / 'job.sh'


def train(args):
    # parse model-index.yml
    model_index_file = MMAGIC_ROOT / 'model-index.yml'

    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    script_name = osp.join('tools', 'train.py')
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

    preview_script = ''
    pbar = tqdm(models.values())
    for model_info in pbar:

        if model_info.results is None:
            continue

        model_name = model_info.name
        pbar.set_description(model_name)
        if 'cvt' in model_name:
            print(f'Skip converted config: {model_name} ({model_info.config})')
            continue
        script_path = create_train_job_batch(commands, model_info, args, port,
                                             script_name)
        if script_path is not None:
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
        history_log = datetime.now().strftime('train-%Y%m%d-%H%M%S') + '.log'
        with open(history_log, 'w') as fp:
            for job in job_name_list:
                fp.write(job + '\n')
        fp.close()
        print(f'Have saved job submission history in {history_log}')

    else:
        console.print('Please set "--run" to start the job')


def show_summary(summary_data, models_map, work_dir, save=False):
    table = Table(title='Train Benchmark Regression Summary')
    table.add_column('Model')
    md_header = ['Model']
    for metric in METRICS_MAP:
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
        for metric_key in METRICS_MAP:
            if metric_key in summary:
                metric = summary[metric_key]
                expect = round(metric['expect'], 2)
                result = round(metric['result'], 2)
                tolerance = metric['tolerance']
                rule = metric['rule']
                color = set_color(result, expect, tolerance, rule)
                row.append(f'{expect:.2f}')
                row.append(f'[{color}]{result:.2f}[/{color}]')
                md_row.append(f'{expect:.2f}')
                md_row.append(f'{result:.2f}')
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
        summary_path = work_dir / 'train_benchmark_summary.md'
        with open(summary_path, 'w') as file:
            file.write('# Train Benchmark Regression Summary\n')
            file.writelines(md_rows)


def summary(args):
    model_index_file = MMAGIC_ROOT / 'model-index.yml'
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
        for key_yml, key_tolerance in METRICS_MAP.items():
            key_results = key_tolerance['keys']
            tolerance = key_tolerance['tolerance']
            rule = key_tolerance['rule']

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

    show_summary(summary_data, models, work_dir, args.save)
    # if args.save:
    #     save_summary(summary_data, models, work_dir)


def main():
    args = parse_args()

    if args.summary:
        summary(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
