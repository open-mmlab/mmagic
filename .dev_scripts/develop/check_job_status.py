import argparse
import os
import os.path as osp
from glob import glob

from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='check srun job status and rerun failed jobs')
    parser.add_argument(
        '--work_dirs', type=str, default='work_dirs/benchmark_train')
    parser.add_argument(
        '--type',
        type=str,
        default='failed',
        choices=['running', 'success', 'queue', 'failed', 'all'],
    )
    args = parser.parse_args()

    models = os.listdir(args.work_dirs)

    failed_list = []
    queue_list = []
    running_list = []

    for m in tqdm(models):
        log_file = glob(osp.join(args.work_dirs, m, '*.out'))

        if len(log_file) == 0:
            queue_list.append(osp.join(args.work_dirs, m, 'job.sh'))
        elif len(log_file) >= 1:

            log_file = log_file[0]
            with open(log_file, 'r') as f:
                last_line = f.readlines()[-1]

            # failed list
            if last_line.startswith('srun: error:'):
                failed_list.append(log_file)
            else:
                running_list.append(log_file)

    # rerun failed jobs
    with open('benchmark_failed_job.txt', 'w') as f:
        for failed_case in failed_list:
            f.write(f'{failed_case}\n')
    f.close()

    # check running jobs
    with open('benchmark_running_jobs.txt', 'w') as f:
        for running_case in running_list:
            f.write(f'{running_case}\n')
    f.close()

    # check queue jobs
    with open('benchmark_queue_jobs.txt', 'w') as f:
        for queue_case in queue_list:
            f.write(f'{queue_case}\n')
    f.close()
