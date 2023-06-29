from typing import Tuple
from torch.nn import Module

from ....data.typed_graph import TypedGraphLayout
from ...modules.base import AttributeModule
from ...modules.base_types import UpdateModule, UpdateModuleImpl


class UpdateConfigImpl(Module):
    function: UpdateModuleImpl
    residual: bool

    def __init__(self, function: UpdateModuleImpl, residual: bool) -> None:
        super().__init__()

        self.function = function
        self.residual = residual


class UpdateConfig(AttributeModule[UpdateConfigImpl]):
    _update: UpdateModule
    _residual: bool

    def __init__(
        self,
        update: UpdateModule,
        residual: bool,
    ):
        super().__init__()

        self._update = update
        self._residual = residual

    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[UpdateConfigImpl, TypedGraphLayout]:
        function, layout = self._update(node_key, attr_key, layout)

        return (UpdateConfigImpl(function, self._residual), layout)
