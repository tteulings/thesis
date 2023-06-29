from typing import List, Tuple

from torch import Tensor
from torch.nn import Sequential

from ....data.typed_graph import TypedGraphLayout
from ...util import make_mlp
from .base import EdgeEncoder, EncoderImpl, NodeEncoder


class DenseEncoderImpl(EncoderImpl):
    mlp: Sequential

    def __init__(self, sizes: List[int]) -> None:
        super().__init__()

        self.mlp = make_mlp(sizes)

    def forward(self, attr: Tensor) -> Tensor:
        return self.mlp(attr)


class DenseEdgeEncoder(EdgeEncoder):
    _hiddens: List[int]

    def __init__(self, hiddens: List[int]) -> None:
        super().__init__()

        self._hiddens = hiddens

    def __call__(
        self, edge_key: str, layout: TypedGraphLayout
    ) -> Tuple[EncoderImpl, TypedGraphLayout]:
        input_size = layout.edge_sets[edge_key].attrs

        layout.edge_sets[edge_key].attrs = self._hiddens[-1]

        return (DenseEncoderImpl([input_size] + self._hiddens), layout)


class DenseNodeEncoder(NodeEncoder):
    _hiddens: List[int]

    def __init__(self, hiddens: List[int]) -> None:
        super().__init__()

        self._hiddens = hiddens

    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[EncoderImpl, TypedGraphLayout]:
        input_size = layout.node_sets[node_key].attrs[attr_key]

        layout.node_sets[node_key].attrs[attr_key] = self._hiddens[-1]

        return (DenseEncoderImpl([input_size] + self._hiddens), layout)
