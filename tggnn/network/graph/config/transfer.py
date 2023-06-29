from typing import Tuple
from torch.nn import Module

from ....data.typed_graph import TypedGraphLayout
from ...modules.base import SetModule
from ...modules.base_types import TransferModule, TransferModuleImpl


class TransferConfigImpl(Module):
    function: TransferModuleImpl
    residual: bool

    def __init__(
        self,
        transfer: TransferModuleImpl,
        residual: bool,
    ):
        super().__init__()

        self.function = transfer
        self.residual = residual


class TransferConfig(SetModule[TransferConfigImpl]):
    _transfer: TransferModule
    _residual: bool

    def __init__(
        self,
        transfer: TransferModule,
        residual: bool,
    ):
        super().__init__()

        self._transfer = transfer
        self._residual = residual

    def __call__(
        self, edge_key: str, layout: TypedGraphLayout
    ) -> Tuple[TransferConfigImpl, TypedGraphLayout]:
        function, layout = self._transfer(edge_key, layout)

        return (TransferConfigImpl(function, self._residual), layout)
