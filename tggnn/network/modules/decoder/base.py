import abc

from torch import Tensor
from torch.nn import Module

from ..base import SetModule, AttributeModule


class DecoderImpl(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, attr: Tensor) -> Tensor:
        pass


EdgeDecoder = SetModule[DecoderImpl]
NodeDecoder = AttributeModule[DecoderImpl]
