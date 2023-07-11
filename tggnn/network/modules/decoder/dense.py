from typing import List, Tuple

from torch import Tensor
from torch.nn import Sequential

from ....data.typed_graph.typed_graph import TypedGraphLayout
from ...util import make_mlp
from .base import DecoderImpl, EdgeDecoder, NodeDecoder


class DenseDecoderImpl(DecoderImpl):
    mlp: Sequential

    def __init__(self, layout: List[int]) -> None:
        super().__init__()

        self.mlp = make_mlp(layout, activate_final=False, normalize=False)

    def forward(self, attr: Tensor) -> Tensor:
        return self.mlp(attr)


class DenseEdgeDecoder(EdgeDecoder):
    _hiddens: List[int]
    _label_key: str

    def __init__(self, hiddens: List[int], label_key: str) -> None:
        super().__init__()

        self._hiddens = hiddens
        self._label_key = label_key

    def __call__(
        self, edge_key: str, layout: TypedGraphLayout
    ) -> Tuple[DecoderImpl, TypedGraphLayout]:
        input_size = layout.edge_sets[edge_key].attrs
        output_size = layout.labels[self._label_key].attrs

        layout.edge_sets[edge_key].attrs = output_size

        return (
            DenseDecoderImpl([input_size] + self._hiddens + [output_size]),
            layout,
        )


class DenseNodeDecoder(NodeDecoder):
    _hiddens: List[int]
    _label_key: str

    def __init__(self, hiddens: List[int], label_key: str) -> None:
        super().__init__()

        self._hiddens = hiddens
        self._label_key = label_key

    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[DecoderImpl, TypedGraphLayout]:
        input_size = layout.node_sets[node_key].attrs[attr_key]
        output_size = layout.labels[self._label_key].attrs

        layout.node_sets[node_key].attrs[attr_key] = output_size

        return (
            DenseDecoderImpl([input_size] + self._hiddens + [output_size]),
            layout,
        )


class DenseCentroidDecoder(NodeDecoder):
    _hiddens: List[int]
    _label_key: str

    def __init__(self, hiddens: List[int], label_key: str) -> None:
        super().__init__()

        self._hiddens = hiddens
        self._label_key = label_key

    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[DecoderImpl, TypedGraphLayout]:
        input_size = layout.node_sets[node_key].attrs[attr_key] + layout.node_sets['centroid'].attrs['memory']
        output_size = layout.labels[self._label_key].attrs

        layout.node_sets[node_key].attrs[attr_key] = output_size

        return (
            DenseDecoderImpl([input_size] + self._hiddens + [output_size]),
            layout,
        )