import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod

def size(p):
    return np.prod(p.size())

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        torch.nn.Module.__init__(self)

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    @property
    def num_parameters(self):
        return sum([size(param) for param in self.parameters()])

    def get_f(self, name):
        raise NotImplementedError()

    def get_num_cell_parameters(self, dag):
        raise NotImplementedError()

    def reset_parameters(self):
        raise NotImplementedError()