from typing import Tuple

import torch
from torch import Tensor

from ....data.typed_graph import TypedGraphLayout
from .base import EdgeEncoder, EncoderImpl, NodeEncoder


class NullEncoderImpl(EncoderImpl):
    latent_size: int

    def __init__(self, latent_size: int) -> None:
        super().__init__()

        self.latent_size = latent_size

    def forward(self, attr: Tensor) -> Tensor:
        return torch.zeros((attr.size()[0], self.latent_size))


class NullEdgeEncoder(EdgeEncoder):
    _latent_size: int

    def __init__(self, latent_size: int) -> None:
        super().__init__()

        self._latent_size = latent_size

    def __call__(
        self, edge_key: str, layout: TypedGraphLayout
    ) -> Tuple[EncoderImpl, TypedGraphLayout]:
        layout.edge_sets[edge_key].attrs = self._latent_size

        return (NullEncoderImpl(self._latent_size), layout)


class NullNodeEncoder(NodeEncoder):
    _latent_size: int

    def __init__(self, latent_size: int) -> None:
        super().__init__()

        self._latent_size = latent_size

    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[EncoderImpl, TypedGraphLayout]:
        layout.node_sets[node_key].attrs[attr_key] = self._latent_size

        return (NullEncoderImpl(self._latent_size), layout)
