import torch
import torch.nn as nn


class EltwiseAdd(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise addition"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res


class EltwiseSub(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise subtraction"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res -= t
        else:
            for t in input[1:]:
                res = res - t
        return res


class EltwiseMult(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise multiplication"""
        super().__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res *= t
        else:
            for t in input[1:]:
                res = res * t
        return res


class EltwiseDiv(nn.Module):
    def __init__(self, inplace=False):
        """Element-wise division"""
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor, y):
        if self.inplace:
            return x.div_(y)
        return x.div(y)
