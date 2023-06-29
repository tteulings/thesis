from typing import Dict, List, Tuple

from torch import Tensor
from torch.types import Device

from ...common.util.aggregate import Aggregate
from ..abstract import Index
from ..typed_graph import NodeSet


class PassthroughIndex(Index):
    def select(
        self,
        source: NodeSet,
        source_keys: List[str],
        target: NodeSet,
        target_keys: List[str],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        dims = [source[key].num_nodes() for key in source_keys] + [
            target[key].num_nodes() for key in target_keys
        ]

        if any(dim != dims[0] for dim in dims):
            raise Exception(
                "PassthroughIndex: All sources and targets must have equal"
                f" node dimension (sources: {source_keys}, targets:"
                f" {target_keys})."
            )

        return (
            {key: source[key].attr for key in source_keys},
            {key: target[key].attr for key in target_keys},
        )

    def select_single(
        self, source: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # TODO: Add dimensionality check here?
        return (source, target)

    def gather(self, edge_attr: Tensor, _: Aggregate) -> Tensor:
        return edge_attr

    def to(self, _: Device) -> None:
        pass
