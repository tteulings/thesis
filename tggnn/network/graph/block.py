from typing import Generic, Tuple, TypeVar

from ...data.typed_graph import TypedGraph, TypedGraphLayout
from ..container import ModuleList
from .config.block import GraphBlockConfig
from .module import GraphModule
from .passes import GraphPass, GraphPassBase

TG_Data = TypeVar("TG_Data", bound=TypedGraph)


class GraphBlockImpl(GraphModule[TG_Data], Generic[TG_Data]):
    _passes: ModuleList[GraphPassBase[TG_Data]]

    def __init__(self, passes: ModuleList[GraphPassBase[TG_Data]]) -> None:
        super().__init__()

        self._passes = passes

    def forward(self, data: TG_Data) -> TG_Data:
        for graph_pass in self._passes:
            data = graph_pass.forward(data)

        return data


class GraphBlock(Generic[TG_Data]):
    @staticmethod
    def __call__(
        config: GraphBlockConfig, layout: TypedGraphLayout
    ) -> Tuple[GraphBlockImpl[TG_Data], TypedGraphLayout]:
        passes = ModuleList()

        for _ in range(config.iterations):
            graph_pass, layout = GraphPass[TG_Data]()(
                config.graph_config, layout
            )
            passes.append(graph_pass)

        return (GraphBlockImpl(passes), layout)
