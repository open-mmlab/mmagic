import os
import os.path as osp
from typing import Tuple

from rich import print as pprint
from rich.table import Table


def parse_job_list(job_list) -> Tuple[list, list]:
    """Parse task name and job id from list. All elements in `job_list` must.

    be formatted as `JOBID @ JOBNAME`.

    Args:
        job_list (list[str]): Job list.

    Returns:
        Tuple[list, list]: Job ID list and Job name list.
    """
    assert all([
        ' @ ' in job for job in job_list
    ]), ('Each line of job list must be formatted like \'JOBID @ JOBNAME\'.')
    job_id_list, job_name_list = [], []
    for job_info in job_list:
        job_id, job_name = job_info.split(' @ ')
        job_id_list.append(job_id)
        job_name_list.append(job_name)
    return job_id_list, job_name_list


def parse_job_list_from_file(job_list_file: str) -> Tuple[list, list]:
    """Parse job list from file and return a tuple contains list of job id and
    job name.

    Args:
        job_list_file (str): The path to the file list.

    Returns:
        Tuple[list, list]: A tuple contains list of job id and job name.
    """
    if not osp.exists(job_list_file):
        return False
    with open(job_list_file, 'r') as file:
        job_list = [job.strip() for job in file.readlines()]
    return parse_job_list(job_list)


def get_info_from_id(job_id: str) -> dict:
    """Get the basic information of a job id with `swatch examine` command.

    Args:
        job_id (str): The ID of the job.

    Returns:
        dict: A dict contains information of the corresponding job id.
    """
    # NOTE: do not have exception handling here
    info_stream = os.popen(f'swatch examine {job_id}')
    info_str = [line.strip() for line in info_stream.readlines()]
    status_info = info_str[2].split()
    try:
        status_dict = {
            'JobID': status_info[0],
            'JobName': status_info[1],
            'Partition': status_info[2],
            'NNodes': status_info[3],
            'AllocCPUS': status_info[4],
            'State': status_info[5]
        }
    except Exception:
        print(job_id)
        print(info_str)
    return status_dict


def filter_jobs(job_id_list: list,
                job_name_list: list,
                select: list = ['FAILED'],
                show_table: bool = False,
                table_name: str = 'Filter Results') -> Tuple[list, list]:
    """Filter the job which status not belong to :attr:`select`.

    Args:
        job_id_list (list): The list of job ids.
        job_name_list (list): The list of job names.
        select (list, optional): Which kind of jobs will be selected.
            Defaults to ['FAILED'].
        show_table (bool, optional): Whether display the filter result in a
            table. Defaults to False.
        table_name (str, optional): The name of the table. Defaults to
            'Filter Results'.

    Returns:
        Tuple[list]: A tuple contains selected job ids and job names.
    """
    # if ignore is not passed, return the original id list and name list
    if not select:
        return job_id_list, job_name_list
    filtered_id_list, filtered_name_list = [], []
    job_info_list = []
    for id_, name_ in zip(job_id_list, job_name_list):
        info = get_info_from_id(id_)
        job_info_list.append(info)
        if info['State'] in select:
            filtered_id_list.append(id_)
            filtered_name_list.append(name_)

    if show_table:
        filter_table = Table(title=table_name)
        for field in ['Name', 'ID', 'State', 'Is Selected']:
            filter_table.add_column(field)
        for id_, name_, info_ in zip(job_id_list, job_name_list,
                                     job_info_list):
            selected = '[green]True' \
                if info_['State'] in select else '[red]False'
            filter_table.add_row(name_, id_, info_['State'], selected)
        pprint(filter_table)
    return filtered_id_list, filtered_name_list
