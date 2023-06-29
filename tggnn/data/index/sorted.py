from typing import Dict, List, Tuple

from torch import Tensor
from torch.types import Device

from torch_scatter import segment_coo

from ...common.util.aggregate import Aggregate
from ..abstract import Index
from ..typed_graph import NodeSet


class SortedIndex(Index):
    _index: Tensor

    def __init__(self, index: Tensor) -> None:
        super().__init__()

        self._index = index

    def select(
        self,
        source: NodeSet,
        source_keys: List[str],
        target: NodeSet,
        target_keys: List[str],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        source_idx, target_idx = self._index
        return (
            {
                attr_key: source[attr_key].attr.index_select(0, source_idx)
                for attr_key in source_keys
            },
            {
                attr_key: target[attr_key].attr.index_select(0, target_idx)
                for attr_key in target_keys
            },
        )

    def select_single(
        self, source: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        source_idx, target_idx = self._index

        return (
            source.index_select(0, source_idx),
            target.index_select(0, target_idx),
        )

    def gather(self, edge_attr: Tensor, aggr: Aggregate) -> Tensor:
        return segment_coo(
            edge_attr,
            self._index[1],
            reduce=aggr,
        )

    def to(self, device: Device) -> None:
        self._index = self._index.to(device)
