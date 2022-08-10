import argparse

from job_watcher import show_job_status

parser = argparse.ArgumentParser(
    description='check srun job status and rerun failed jobs')
parser.add_argument('--list', type=str, required=True)
parser.add_argument(
    '--work_dirs', type=str, default='work_dirs/benchmark_train')
parser.add_argument(
    '--type',
    type=str,
    default='failed',
    choices=['running', 'success', 'queue', 'failed', 'all'],
)
args = parser.parse_args()

if __name__ == '__main__':
    f = open(args.list, 'r')
    job_name_list = f.readlines()
    plain_txt = show_job_status(args.work_dirs, job_name_list)
    with open('status.log', 'w') as f:
        f.write(plain_txt)
