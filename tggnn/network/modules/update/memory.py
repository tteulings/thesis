from typing import Iterable, Tuple

import torch
from torch import Tensor
from torch.nn import GRUCell

from ....data.typed_graph import TypedGraphLayout
from .base import UpdateModule, UpdateModuleImpl


class MemoryUpdateImpl(UpdateModuleImpl):
    _gru: GRUCell

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()

        self._gru = GRUCell(input_size=input_size, hidden_size=hidden_size)

    def forward(
        self,
        node_attr: Tensor,
        messages: Iterable[Tensor],
        center_velocity: Tensor,
    ) -> Tensor:
        print(center_velocity.shape)
        

        return self._gru.forward(torch.cat([*messages, center_velocity], 1), node_attr)


class MemoryUpdate(UpdateModule):
    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[UpdateModuleImpl, TypedGraphLayout]:
        node_set = layout.node_sets[node_key]
 
        return (
            MemoryUpdateImpl(
                2*sum(
                    (
                        layout.edge_sets[edge_key].attrs
                        for edge_key in node_set.edge_sets
                    )
                ),
                node_set.attrs[attr_key],
            ),
            layout,
        )
