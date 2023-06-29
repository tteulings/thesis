from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Sequential

from tggnn.data.typed_graph import TypedGraphLayout
from tggnn.network.modules.base_types import TransferModule, TransferModuleImpl
from tggnn.network.util.mlp import make_mlp


class EGCLTransferImpl(TransferModuleImpl):
    _mlp: Sequential
    _pos: Tuple[str, str]
    _source_keys: List[str]
    _target_keys: List[str]

    def __init__(
        self,
        layout: List[int],
        pos: Tuple[str, str],
        source_keys: List[str],
        target_keys: List[str],
    ):
        super().__init__()

        self._pos = pos
        self._source_keys = source_keys
        self._target_keys = target_keys

        self._mlp = make_mlp(layout, True, False)

    def forward(
        self,
        source: Dict[str, Tensor],
        target: Dict[str, Tensor],
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        source_pos, target_pos = self._pos

        return self._mlp.forward(
            torch.cat(
                (
                    *source.values(),
                    *target.values(),
                    torch.norm(
                        target[target_pos] - source[source_pos],
                        dim=1,
                        keepdim=1,
                    ),
                    *([edge_attr] if edge_attr is not None else []),
                ),
                dim=1,
            )
        )

    def attribute_keys(self) -> Tuple[List[str], List[str]]:
        source_pos, target_pos = self._pos

        return (
            [source_pos] + self._source_keys,
            [target_pos] + self._target_keys,
        )


class EGCLTransfer(TransferModule):
    _hiddens: List[int]
    _pos: Tuple[str, str]
    _source_keys: List[str]
    _target_keys: List[str]

    def __init__(
        self,
        hiddens: List[int],
        pos: Tuple[str, str],
        source_keys: List[str],
        target_keys: List[str],
    ) -> None:
        super().__init__()

        self._hiddens = hiddens
        self._pos = pos
        self._source_keys = source_keys
        self._target_keys = target_keys

    def __call__(
        self, edge_key: str, layout: TypedGraphLayout
    ) -> Tuple[TransferModuleImpl, TypedGraphLayout]:
        source_pos_key, target_pos_key = self._pos

        edge_set = layout.edge_sets[edge_key]
        source = layout.node_sets[edge_set.source]
        target = layout.node_sets[edge_set.target]

        input_size = (
            source.attrs[source_pos_key]
            + target.attrs[target_pos_key]
            + sum(source.attrs[key] for key in self._source_keys)
            + sum(target.attrs[key] for key in self._source_keys)
            + edge_set.attrs
        )

        edge_set.attrs = self._hiddens[-1]

        return (
            EGCLTransferImpl(
                [input_size] + self._hiddens,
                self._pos,
                self._source_keys,
                self._target_keys,
            ),
            layout,
        )
