from typing import Generic, TypeVar

from ....data.typed_graph import TypedGraph
from ...container import ModuleDict
from ..config import TransferConfigImpl
from ..module import GraphModule

TG_Data = TypeVar("TG_Data", bound=TypedGraph)


class EdgePass(GraphModule[TG_Data], Generic[TG_Data]):
    _configs: ModuleDict[TransferConfigImpl]

    def __init__(self, transfer_configs: ModuleDict[TransferConfigImpl]):
        super().__init__()

        self._configs = transfer_configs

    def forward(self, data: TG_Data) -> TG_Data:
        for name, config in self._configs.items():
            edge_set = data.edge_sets[name]

            source_keys, target_keys = config.function.attribute_keys()

            source, target = edge_set.index.select(
                data.node_sets[edge_set.source],
                source_keys,
                data.node_sets[edge_set.target],
                target_keys,
            )

            attr = config.function(source, target, edge_set.attr)

            edge_set.attr = (
                edge_set.attr + attr
                if config.residual and edge_set.attr is not None
                else attr
            )

        return data
