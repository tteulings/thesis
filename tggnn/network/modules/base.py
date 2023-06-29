import abc
from typing import Generic, Tuple, TypeVar

from torch.nn import Module

from ...data.typed_graph import TypedGraphLayout


T_Module = TypeVar("T_Module", bound=Module)


class SetModule(Generic[T_Module], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self, edge_key: str, layout: TypedGraphLayout
    ) -> Tuple[T_Module, TypedGraphLayout]:
        pass


class AttributeModule(Generic[T_Module], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[T_Module, TypedGraphLayout]:
        pass
