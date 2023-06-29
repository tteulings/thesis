from typing import Iterable, List, Tuple

import torch
from torch import Tensor
from torch.nn import Sequential

from ....data.typed_graph import TypedGraphLayout
from ...util import make_mlp
from .base import UpdateModuleImpl, UpdateModule


class DenseUpdateImpl(UpdateModuleImpl):
    mlp: Sequential

    def __init__(
        self,
        layout: List[int],
        activate_final: bool,
        normalize: bool,
    ) -> None:
        super().__init__()

        self.mlp = make_mlp(layout, activate_final, normalize)

    def forward(
        self,
        node_attr: Tensor,
        messages: Iterable[Tensor],
    ) -> Tensor:
        return self.mlp(torch.cat([node_attr, *messages], 1))


class DenseUpdate(UpdateModule):
    _layout: List[int]
    _activate_final: bool
    _normalize: bool

    def __init__(
        self,
        layout: List[int],
        activate_final: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self._layout = layout
        self._activate_final = activate_final
        self._normalize = normalize

    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[UpdateModuleImpl, TypedGraphLayout]:
        node_set = layout.node_sets[node_key]
        node_attrs = node_set.attrs[attr_key]

        input_size = node_attrs + sum(
            (
                layout.edge_sets[edge_key].attrs
                for edge_key in node_set.edge_sets
            )
        )

        node_set.attrs[attr_key] = self._layout[-1]

        return (
            DenseUpdateImpl(
                [input_size] + self._layout,
                self._activate_final,
                self._normalize,
            ),
            layout,
        )
