from typing import Generic, Optional, Tuple, TypeVar

from ....data.typed_graph import TypedGraph, TypedGraphLayout
from ...container.module_dict import ModuleDict
from ...util import AttributeModuleDict, SetModuleDict
from ..config import (
    AggregateConfig,
    GraphPassConfig,
    TransferConfigImpl,
    UpdateConfigImpl,
)
from ..module import GraphModule
from .edge import EdgePass
from .node import NodePass

TG_Data = TypeVar("TG_Data", bound=TypedGraph)


class GraphPassBase(GraphModule[TG_Data], Generic[TG_Data]):
    def __init__(
        self,
        edge_pass: Optional[GraphModule[TG_Data]] = None,
        node_pass: Optional[GraphModule[TG_Data]] = None,
    ):
        super().__init__()

        self._edge_pass = edge_pass
        self._node_pass = node_pass

    def forward(self, data: TG_Data) -> TG_Data:
        if self._edge_pass is not None:
            data = self._edge_pass.forward(data)

        if self._node_pass is not None:
            data = self._node_pass.forward(data)

        return data


class GraphPassImpl(GraphPassBase[TG_Data], Generic[TG_Data]):
    def __init__(
        self,
        update_configs: ModuleDict[ModuleDict[UpdateConfigImpl]],
        transfer_configs: ModuleDict[TransferConfigImpl],
        aggregate_configs: ModuleDict[AggregateConfig],
    ):
        super().__init__(
            EdgePass(transfer_configs),
            NodePass(update_configs, aggregate_configs),
        )


class GraphPass(Generic[TG_Data]):
    @staticmethod
    def __call__(
        config: GraphPassConfig,
        layout: TypedGraphLayout,
    ) -> Tuple[GraphPassImpl[TG_Data], TypedGraphLayout]:
        transfer_modules, layout = SetModuleDict(config.transfers.items()).init(
            layout
        )

        update_modules, layout = AttributeModuleDict(
            (
                (node_key, attr_updates.items())
                for node_key, attr_updates in config.updates.items()
            )
        ).init(layout)

        aggregate_modules = ModuleDict(config.aggregates)

        return (
            GraphPassImpl[TG_Data](
                update_modules, transfer_modules, aggregate_modules
            ),
            layout,
        )
