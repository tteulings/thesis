from typing import Generic, Iterable, Tuple, TypeVar

from torch.nn import Module

from ...data.typed_graph import TypedGraphLayout
from ..container import ModuleDict
from ..modules.base import SetModule, AttributeModule

T_Module = TypeVar("T_Module", bound=Module)


class SetModuleDict(Generic[T_Module]):
    def __init__(
        self,
        key_fn_pairs: Iterable[Tuple[str, SetModule[T_Module]]],
    ) -> None:
        self._key_fn_pairs = key_fn_pairs

    def init(
        self, layout: TypedGraphLayout
    ) -> Tuple[ModuleDict[T_Module], TypedGraphLayout]:
        module_dict: ModuleDict[T_Module] = ModuleDict()

        for key, lazy_module in self._key_fn_pairs:
            module, layout = lazy_module(key, layout)
            module_dict.add_module(key, module)

        return (module_dict, layout)


class AttributeModuleDict(Generic[T_Module]):
    def __init__(
        self,
        nested_key_fn_pairs: Iterable[
            Tuple[str, Iterable[Tuple[str, AttributeModule[T_Module]]]]
        ],
    ) -> None:
        self._node_set_modules = nested_key_fn_pairs

    def init(
        self, layout: TypedGraphLayout
    ) -> Tuple[ModuleDict[ModuleDict[T_Module]], TypedGraphLayout]:
        node_set_dict: ModuleDict[ModuleDict[T_Module]] = ModuleDict()

        for node_key, attr_modules in self._node_set_modules:
            attr_module_dict = ModuleDict()

            for attr_key, lazy_module in attr_modules:
                module, layout = lazy_module(node_key, attr_key, layout)
                attr_module_dict.add_module(attr_key, module)

            node_set_dict.add_module(node_key, attr_module_dict)

        return (node_set_dict, layout)
