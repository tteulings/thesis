from typing import Tuple

from torch import Tensor

from ....data.typed_graph import TypedGraphLayout
from .base import EdgeEncoder, EncoderImpl, NodeEncoder


class NoOpEncoderImpl(EncoderImpl):
    def forward(self, attr: Tensor) -> Tensor:
        # if x.size()[1] != self.latent_size:
        #     raise Exception(
        #         "Applying NoOpEncoder to a set with number of attributes "
        #         "unequal to the latent size."
        #     )

        return attr


class NoOpEdgeEncoder(EdgeEncoder):
    def __call__(
        self, edge_key: str, layout: TypedGraphLayout
    ) -> Tuple[EncoderImpl, TypedGraphLayout]:
        return (NoOpEncoderImpl(), layout)


class NoOpNodeEncoder(NodeEncoder):
    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[EncoderImpl, TypedGraphLayout]:
        return (NoOpEncoderImpl(), layout)
