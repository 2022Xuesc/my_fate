import numpy as np
import torch
from scipy import sparse
import federatedml.nn.backend.distiller.utils as utils


class CompressedTensor:
    def __init__(self, data, indices=None, indptr=None, shape=None):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape


def get_csr_arr_size(csr_arr):
    return csr_arr.data.nbytes + csr_arr.indices.nbytes + csr_arr.indptr.nbytes


# Todo: 根据稀疏水平直接决定是否采用压缩方案

# 目前是,如果采用稀疏方案，则
def compress_tensor(tensors):
    tensors_bytes = 0
    compressed_tensors = []
    for tensor in tensors:
        # 如果是卷积层或者全连接层的参数，则进行压缩
        shape = tensor.shape
        nnz = utils.nnz(tensor)
        if len(shape) > 1 and nnz < (shape[0] * (np.prod(shape[1:]) - 1) - 1) / 2:
            arr = tensor.view(shape[0], -1).data.cpu().numpy()
            # 进行压缩
            csr_arr = sparse.csr_matrix(arr)
            tensors_bytes += get_csr_arr_size(csr_arr)
            compressed_tensors.append(
                CompressedTensor(csr_arr.data, csr_arr.indices, csr_arr.indptr, shape))
        else:
            arr = tensor.data.cpu().numpy()
            tensors_bytes += arr.nbytes
            compressed_tensors.append(CompressedTensor(arr))
    return compressed_tensors, tensors_bytes


def restore_tensor(compressed_tensors):
    tensors_bytes = 0
    tensors = []
    for compressed_tensor in compressed_tensors:
        shape = compressed_tensor.shape
        if shape is None:
            tensors.append(compressed_tensor.data)
            tensors_bytes += compressed_tensor.data.nbytes
        else:
            csr_arr = sparse.csr_matrix(
                (compressed_tensor.data, compressed_tensor.indices, compressed_tensor.indptr), shape=shape)
            tensors_bytes += get_csr_arr_size(csr_arr)
            tensors.append(torch.from_numpy(csr_arr.toarray()).expand(shape[0], np.prod(shape[1:])).view(*shape).numpy())
    return tensors,tensors_bytes