import torch

def truncated_normal(tensor, mean=0, std=1, n_truncted_stds=2):
    assert std >= 0, f"{std}"
    size = tensor.shape

    # [n, 1] -> [n, 1, 4]
    tmp = tensor.new_empty(size + (4,), device=tensor.device).normal_()
    tmp.data.mul_(std).add_(mean)

    lower_bound = mean - 1 * n_truncted_stds * std
    upper_bound = mean + n_truncted_stds * std
    valid = (tmp < upper_bound) & (tmp > lower_bound)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))

    try:
        assert torch.all(tensor >= lower_bound), f"{torch.min(tensor)}"
        assert torch.all(tensor <= upper_bound), f"{torch.max(tensor)}"
    except:
        # fmt: off
        print("\nin truncated normal lower bound: ", tensor.shape, lower_bound, torch.min(tensor), torch.sum(tensor >= lower_bound))
        print("\nin truncated normal upper bound: ", tensor.shape, upper_bound, torch.max(tensor), torch.sum(tensor <= lower_bound))
        tensor[tensor <= lower_bound] = lower_bound
        tensor[tensor >= upper_bound] = upper_bound
        # fmt: on

    return tensor


def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)