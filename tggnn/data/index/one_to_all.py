from typing import Dict, List, Tuple

from torch import Tensor
from torch.types import Device

from ...common.util.aggregate import Aggregate
from ..abstract import Index
from ..typed_graph import NodeSet


class OneToAllIndex(Index):
    def select(
        self,
        source: NodeSet,
        source_keys: List[str],
        target: NodeSet,
        target_keys: List[str],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        if len(target_keys) == 0:
            raise Exception(
                "No target specified for OneToAllIndex. Cannot infer target"
                f" dimension (sources: {source_keys})."
            )

        num_nodes = target[target_keys[0]].num_nodes()

        if any(target[key].num_nodes() != num_nodes for key in target_keys):
            raise Exception(
                "OneToAllIndex: Target dimensions are not equal for all targets"
                f" ({target_keys})."
            )
        if any(source[key].num_nodes() != 1 for key in source_keys):
            raise Exception(
                "OneToAllIndex: Source dimensions are not equal to one for all"
                f" sources ({source_keys})."
            )

        return (
            {
                key: source[key].attr.expand(num_nodes, -1)
                for key in source_keys
            },
            {key: target[key].attr for key in target_keys},
        )

    def select_single(
        self, source: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return (source.expand_as(target), source)

    def gather(self, edge_attr: Tensor, _: Aggregate) -> Tensor:
        return edge_attr

    def to(self, _: Device) -> None:
        pass
