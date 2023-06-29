from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.types import Device

from ...common.util.aggregate import Aggregate
from ..abstract import Index
from ..typed_graph import NodeSet


class AllToOneIndex(Index):
    def select(
        self,
        source: NodeSet,
        source_keys: List[str],
        target: NodeSet,
        target_keys: List[str],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        if len(source_keys) == 0:
            raise Exception(
                "No source specified for AllToOneIndex. Cannot infer source"
                f" dimension (targets: {target_keys})."
            )

        num_nodes = source[source_keys[0]].num_nodes()

        if any(source[key].num_nodes() != num_nodes for key in source_keys):
            raise Exception(
                "OneToAllIndex: Source dimensions are not equal for all sources"
                f" ({source_keys})."
            )
        if any(target[key].num_nodes() != 1 for key in target_keys):
            raise Exception(
                "OneToAllIndex: Target dimensions are not equal to one for all"
                f" targets ({target_keys})."
            )

        return (
            {key: source[key].attr for key in source_keys},
            {
                key: target[key].attr.expand(num_nodes, -1)
                for key in target_keys
            },
        )

    def select_single(
        self, source: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return (source, target.expand_as(source))

    def gather(self, edge_attr: Tensor, aggr: Aggregate) -> Tensor:
        return {
            Aggregate.SUM: torch.sum(edge_attr, dim=0),
            Aggregate.MUL: torch.prod(edge_attr, dim=0),
            Aggregate.MEAN: torch.mean(edge_attr, dim=0),
            Aggregate.MIN: torch.min(edge_attr, dim=0),
            Aggregate.MAX: torch.max(edge_attr, dim=0),
        }[aggr].unsqueeze(dim=0)

    def to(self, _: Device) -> None:
        pass
