from typing import Generic, TypeVar

from ....data.typed_graph import TypedGraph
from ...container import ModuleDict
from ..config import AggregateConfig, UpdateConfigImpl
from ..module import GraphModule

TG_Data = TypeVar("TG_Data", bound=TypedGraph)


class NodePass(GraphModule[TG_Data], Generic[TG_Data]):
    _update_configs: ModuleDict[ModuleDict[UpdateConfigImpl]]
    _aggregate_configs: ModuleDict[AggregateConfig]

    def __init__(
        self,
        update_configs: ModuleDict[ModuleDict[UpdateConfigImpl]],
        aggregate_configs: ModuleDict[AggregateConfig],
    ):
        super().__init__()

        self._update_configs = update_configs
        self._aggregate_configs = aggregate_configs

    def forward(self, data: TG_Data) -> TG_Data:
        for node_key, configs in self._update_configs.items():
            node_set = data.node_sets[node_key]

            messages = (
                edge_set.index.gather(edge_set.attr, aggr)
                for edge_set, aggr in map(
                    lambda edge_key: (
                        data.edge_sets[edge_key],
                        self._aggregate_configs[edge_key].aggregate,
                    ),
                    node_set.edge_sets,
                )
                if edge_set.attr is not None
            )

            for attr_key, config in configs.items():
                attr = config.function(node_set[attr_key].attr, messages)
                node_set[attr_key].attr = (
                    node_set[attr_key].attr + attr if config.residual else attr
                )

        return data
