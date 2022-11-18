import os
import os.path as osp
import time
from argparse import ArgumentParser
from functools import partial

from prettytable import PrettyTable
from pygments import formatters, highlight, lexers
from pygments.util import ClassNotFound
from simple_term_menu import TerminalMenu

CACHE_DIR = osp.join(osp.abspath('~'), '.task_watcher')


def show_job_out(name, root, job_name_list):
    if name == 'Show Status':
        return show_job_status(root, job_name_list)

    job_id, job_name = name.split(' @ ')
    job_out_path = osp.join(root, job_name, f'job.{job_id.strip()}.out')
    try:
        with open(job_out_path, 'r') as file:
            out_content = file.read()
    except Exception:
        out_content = f'{job_out_path} not find.'

    out_content = out_content.split('\n')[-20:]
    out_content = '\n'.join(out_content)

    try:
        lexer = lexers.get_lexer_for_filename(
            job_out_path, stripnl=False, stripall=False)
    except ClassNotFound:
        lexer = lexers.get_lexer_by_name('text', stripnl=False, stripall=False)
    formatter = formatters.TerminalFormatter(bg='dark')  # dark or light
    highlighted_file_content = highlight(out_content, lexer, formatter)
    return highlighted_file_content


def show_job_status(root, job_name_list, csv_path=None):
    """Show job status and dump to csv.

    Args:
        root (_type_): _description_
        job_name_list (_type_): _description_
    Returns:
        _type_: _description_
    """
    table = PrettyTable(title='Job Status')
    table.field_names = ['Name', 'ID', 'Status', 'Output']
    swatch_tmp = 'swatch examine {}'
    if csv_path is None:
        csv_path = 'status.cvs'

    for info in job_name_list:
        id_, name = info.split(' @ ')
        name = name.split('\n')[0].strip()
        proc = os.popen(swatch_tmp.format(id_))
        stdout = proc.read().strip().split('\n')
        job_info = [s for s in stdout[2].split(' ') if s]
        status = job_info[5]

        job_out_path = osp.join(root, name, f'job.{id_.strip()}.out')
        if osp.exists(job_out_path):
            with open(job_out_path, 'r') as file:
                out_content = file.read()
            out_content = out_content.split('\n')
            if len(out_content) > 10:
                out_content = out_content[-7:-1]
            out_content = '\n'.join(out_content)
        else:
            out_content = 'No output currently.'
        table.add_row([name, id_, status, out_content])
    with open(csv_path, 'w') as file:
        file.write(table.get_csv_string())
        print(f'save job status to {csv_path}')
    return table.get_string()


def save_for_resume(root, job_name_list):
    """Save job name and job ID for resume.

    Args:
        root (_type_): _description_
        job_name_list (_type_): _description_
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open(osp.join(CACHE_DIR, timestamp), 'w') as file:
        file.write(f'{root}\n')
        file.writelines([n + '\n' for n in job_name_list])

    with open(osp.join(CACHE_DIR, 'latest'), 'w') as file:
        file.write(f'{root}\n')
        file.writelines([n + '\n' for n in job_name_list])


def resume_from_file(file_path):
    """Resume TUI from file.

    Args:
        file_path (_type_): _description_
    """
    with open((file_path), 'r') as file:
        resume_info = file.readlines()
    resume_info = [info.strip() for info in resume_info]
    root = resume_info[0]
    job_name_list = resume_info[1:]
    show_tui(root, job_name_list)


def start_from_proc(root, proc):
    """Start TUI from proc.

    Args:
        root (_type_): _description_
        proc (_type_): _description_
    """
    std_out = proc.read().strip()
    std_out_list = std_out.split('\n')
    num_tasks = len(std_out_list) // 3
    job_name_list = []
    for idx in range(num_tasks):
        config = std_out_list[3 * idx].strip(' ')
        job_name = config.split('/')[-1].split('.')[0].strip(' ')
        job_id = std_out_list[3 * idx + 2].split(' ')[-1]
        job_name_list.append(f'{job_id} @ {job_name}')

    return job_name_list
    # save_for_resume(root, job_name_list)
    # show_tui(root, job_name_list)


def show_tui(root, job_name_list):
    """Start a interactivate task watcher.
    Args:
        root (str): root for benchmark
        job_name_list (list[str]): List of 'JOBID @ JOBNAME'
    """
    show_func = partial(show_job_out, root=root, job_name_list=job_name_list)
    terminal_menu = TerminalMenu(
        job_name_list + ['Show Status'],
        preview_command=show_func,
        preview_size=0.75)
    terminal_menu.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--list', type=str, default=None)
    parser.add_argument('--resume', default='latest')
    parser.add_argument(
        '--work-dir', type=str, default='work_dirs/benchmark_amp')
    parser.add_argument(
        '--type',
        type=str,
        default='failed',
        choices=['running', 'success', 'queue', 'failed', 'all'],
    )
    args = parser.parse_args()

    if args.list is not None:
        f = open(args.list, 'r')
        job_name_list = f.readlines()
        csv_path = osp.basename(args.list).replace('.log', '.csv')
        plain_txt = show_job_status(args.work_dir, job_name_list, csv_path)
        with open('status.log', 'w') as f:
            f.write(plain_txt)
        print('save status to status.log')
    else:
        if args.resume.upper() == 'LATEST':
            resume_from_file(osp.join(CACHE_DIR, 'latest'))
        else:
            resume_from_file(args.resume)
