import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def single_gpu_test(model,
                    data_loader,
                    save_image=False,
                    save_path=None,
                    iteration=None):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        save_image (bool): Whether save image. Default: False.
        save_path (str): The path to save image. Default: None.
        iteration (int): Iteration number. It is used for the save image name.
            Default: None.

    Returns:
        list: The prediction results.
    """
    if save_image and save_path is None:
        raise ValueError(
            "When 'save_image' is True, you should also set 'save_path'.")

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(
                test_mode=True,
                save_image=save_image,
                save_path=save_path,
                iteration=iteration,
                **data)
        results.append(result)

        # get batch size
        for _, v in data.items():
            if isinstance(v, torch.Tensor):
                batch_size = v.size(0)
                break
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   save_image=False,
                   save_path=None,
                   iteration=None,
                   empty_cache=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        save_image (bool): Whether save image. Default: False.
        save_path (str): The path to save image. Default: None.
        iteration (int): Iteration number. It is used for the save image name.
            Default: None.
        empty_cache (bool): empty cache in every iteration. Default: False.

    Returns:
        list: The prediction results.
    """

    if save_image and save_path is None:
        raise ValueError(
            "When 'save_image' is True, you should also set 'save_path'.")
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(
                test_mode=True,
                save_image=save_image,
                save_path=save_path,
                iteration=iteration,
                **data)
        results.append(result)
        if empty_cache:
            torch.cuda.empty_cache()
        if rank == 0:
            # get batch size
            for _, v in data.items():
                if isinstance(v, torch.Tensor):
                    batch_size = v.size(0)
                    break
            for _ in range(batch_size * world_size):
                prog_bar.update()
    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results in cpu mode.

    It saves the results on different gpus to 'tmpdir' and collects
    them by the rank 0 worker.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None

    Returns:
        list: Ordered results.
    """

    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None

    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
        part_list.append(mmcv.load(part_file))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    # remove tmp dir
    shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results in gpu mode.

    It encodes results to gpu tensors and use gpu communication for results
    collection.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.

    Returns:
        list: Ordered results.
    """

    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank != 0:
        return None

    part_list = []
    for recv, shape in zip(part_recv_list, shape_list):
        part_list.append(pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    return ordered_results
