from typing import Tuple

import torch
from torch import Tensor

from ....data.typed_graph import TypedGraphLayout
from .base import (
    EdgeNormalizer,
    LabelNormalizer,
    NodeNormalizer,
    NormalizerImpl,
)


class WelfordNormalizerImpl(NormalizerImpl):
    _max_accumulations: float
    _std_epsilon: float

    _count: Tensor
    _num_accumulations: Tensor
    _sum: Tensor
    _diff_sum: Tensor

    def __init__(
        self,
        num_features: int,
        max_accumulations: float = 1e6,
        std_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()

        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon

        self.register_buffer(
            "_num_accumulations", torch.zeros(1, dtype=torch.float32)
        )
        self.register_buffer("_count", torch.zeros(1, dtype=torch.float32))
        self.register_buffer(
            "_sum", torch.zeros(num_features, dtype=torch.float32)
        )
        self.register_buffer(
            "_diff_sum", torch.zeros(num_features, dtype=torch.float32)
        )

    def accumulate(self, attr: Tensor) -> None:
        if self._num_accumulations == 0:
            self._count += attr.shape[0]
            self._sum += attr.sum(0)

            self._diff_sum += (attr - self.mean()).square().sum(0)

            self._num_accumulations += 1.0

        elif self._num_accumulations < self._max_accumulations:
            old_mean = self.mean()

            self._count += attr.shape[0]
            self._sum += attr.sum(0)

            self._diff_sum += ((attr - old_mean) * (attr - self.mean())).sum(0)

            self._num_accumulations += 1.0

    def forward(self, attr: Tensor) -> Tensor:
        if self.training:
            self.accumulate(attr)

        return (attr - self.mean()) / self.std()

    def inverse(self, attr: Tensor) -> Tensor:
        return attr * self.std() + self.mean()

    def mean(self) -> Tensor:
        return self._sum / self._count

    def std(self) -> Tensor:
        return torch.sqrt(self._diff_sum / self._count).clamp_min(
            self._std_epsilon
        )


class WelfordNormalizerBase:
    _max_accumulations: float
    _std_epsilon: float

    def __init__(
        self,
        max_accumulations: float = 1e6,
        std_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()

        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon


class WelfordNodeNormalizer(WelfordNormalizerBase, NodeNormalizer):
    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[NormalizerImpl, TypedGraphLayout]:
        return (
            WelfordNormalizerImpl(
                layout.node_sets[node_key].attrs[attr_key],
                self._max_accumulations,
                self._std_epsilon,
            ),
            layout,
        )


class WelfordEdgeNormalizer(WelfordNormalizerBase, EdgeNormalizer):
    def __call__(
        self, edge_key: str, layout: TypedGraphLayout
    ) -> Tuple[NormalizerImpl, TypedGraphLayout]:
        return (
            WelfordNormalizerImpl(
                layout.edge_sets[edge_key].attrs,
                self._max_accumulations,
                self._std_epsilon,
            ),
            layout,
        )


class WelfordLabelNormalizer(WelfordNormalizerBase, LabelNormalizer):
    def __call__(
        self, label_key: str, layout: TypedGraphLayout
    ) -> Tuple[NormalizerImpl, TypedGraphLayout]:
        return (
            WelfordNormalizerImpl(
                layout.labels[label_key].attrs,
                self._max_accumulations,
                self._std_epsilon,
            ),
            layout,
        )
