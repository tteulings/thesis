import abc
from typing import Generic, TypeVar

from torch.nn import Module

from ...data.typed_graph import TypedGraph

TG_Data = TypeVar("TG_Data", bound=TypedGraph)


class GraphModule(Generic[TG_Data], Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, data: TG_Data) -> TG_Data:
        pass
