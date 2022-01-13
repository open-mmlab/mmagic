import torch

from mmedit.utils.testing import dict_to_cuda


def test_dict_to_cuda():
    ori_dict = dict(a=torch.rand(1), b='test')
    cuda_dict = dict_to_cuda(ori_dict)
    assert list(cuda_dict.keys()) == ['a', 'b']
    assert isinstance(cuda_dict['a'], torch.Tensor)
    assert isinstance(cuda_dict['b'], str)
    assert cuda_dict['a'].is_cuda
    assert torch.equal(cuda_dict['a'].cpu(), ori_dict['a'])
    assert cuda_dict['b'] == ori_dict['b']
