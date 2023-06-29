import abc

from torch import Tensor
from torch.nn import Module

from ..base import SetModule, AttributeModule


class EncoderImpl(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, attr: Tensor) -> Tensor:
        pass


EdgeEncoder = SetModule[EncoderImpl]
NodeEncoder = AttributeModule[EncoderImpl]
