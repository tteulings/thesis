import abc
from typing import Iterable

from torch import Tensor
from torch.nn import Module

from ..base import AttributeModule


class UpdateModuleImpl(Module, metaclass=abc.ABCMeta):
    def __call__(
        self,
        node_attr: Tensor,
        messages: Iterable[Tensor],
    ) -> Tensor:
        

        return self.forward(node_attr, messages)

    @abc.abstractmethod
    def forward(
        self,
        node_attr: Tensor,
        messages: Iterable[Tensor],
    ) -> Tensor:
        pass


UpdateModule = AttributeModule[UpdateModuleImpl]
