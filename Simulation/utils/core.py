import abc

import numpy as np
import torch
from torch import nn as nn
import Simulation.utils.pytorch_util as ptu

class PyTorchModule(nn.Module, metaclass=abc.ABCMeta):
    """
    Keeping wrapper around to be a bit more future-proof.
    """
    pass

class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """
    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output

def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return ptu.from_numpy(elem_or_tuple).float()


def elem_or_tuple_to_numpy(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(np_ify(x) for x in elem_or_tuple)
    else:
        return np_ify(elem_or_tuple)

def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == bool:
            yield k, v.astype(int)
        else:
            yield k, v

def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other

def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return ptu.from_numpy(np_array_or_other)
    else:
        return np_array_or_other

def np_to_pytorch_batch(np_batch):
    if isinstance(np_batch, dict):
        return {
            k: _elem_or_tuple_to_variable(x)
            for k, x in _filter_batch(np_batch)
            if x.dtype != np.dtype('O')                     # ignore object (e.g. dictionaries)
        }
    else:
        _elem_or_tuple_to_variable(np_batch)


def identity(x):
    return x


