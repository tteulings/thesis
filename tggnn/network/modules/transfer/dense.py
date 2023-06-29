from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Sequential

from ....data.typed_graph import TypedGraphLayout
from ...util import make_mlp
from .base import TransferModule, TransferModuleImpl


class DenseTransferImpl(TransferModuleImpl):
    _mlp: Sequential
    _source_attrs: List[str]
    _target_attrs: List[str]

    def __init__(
        self,
        source_attrs: List[str],
        target_attrs: List[str],
        layout: List[int],
        activate_final: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self._mlp = make_mlp(layout, activate_final, normalize)
        self._source_attrs = source_attrs
        self._target_attrs = target_attrs

    def forward(
        self,
        source: Dict[str, Tensor],
        target: Dict[str, Tensor],
        edge_attr: Optional[Tensor],
    ) -> Tensor:

        return self._mlp(
            torch.cat(
                [tensor for tensor in source.values()]
                + [tensor for tensor in target.values()]
                + ([edge_attr] if edge_attr is not None else []),
                1,
            )
        )

    def attribute_keys(self) -> Tuple[List[str], List[str]]:
        return (self._source_attrs, self._target_attrs)


class DenseTransfer(TransferModule):
    _hiddens: List[int]
    _activate_final: bool
    _normalize: bool
    _source_attrs: List[str]
    _target_attrs: List[str]

    def __init__(
        self,
        hiddens: List[int],
        source_attrs: List[str],
        target_attrs: List[str],
        activate_final: bool = False,
        normalize: bool = True,
    ) -> None:
        self._hiddens = hiddens
        self._activate_final = activate_final
        self._normalize = normalize
        self._source_attrs = source_attrs
        self._target_attrs = target_attrs

    def __call__(
        self, edge_key: str, layout: TypedGraphLayout
    ) -> Tuple[TransferModuleImpl, TypedGraphLayout]:
        edge_set = layout.edge_sets[edge_key]
        input_size = (
            edge_set.attrs
            + sum(
                layout.node_sets[edge_set.source].attrs[attr_key]
                for attr_key in self._source_attrs
            )
            + sum(
                layout.node_sets[edge_set.target].attrs[attr_key]
                for attr_key in self._target_attrs
            )
        )

        layout.edge_sets[edge_key].attrs = self._hiddens[-1]

        return (
            DenseTransferImpl(
                self._source_attrs,
                self._target_attrs,
                [input_size] + self._hiddens,
                self._activate_final,
                self._normalize,
            ),
            layout,
        )
