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


class NaiveNormalizerImpl(NormalizerImpl):
    _max_accumulations: float
    _std_epsilon: float

    _acc_count: Tensor
    _num_accumulations: Tensor
    _acc_sum: Tensor
    _acc_sum_squared: Tensor

    def __init__(
        self,
        num_features: int,
        max_accumulations: float = 1e6,
        std_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()

        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon

        # NOTE: For some reason python specifies the types of self._acc_count
        # and _num_accumulations as 'int', while it should be 'Tensor'.
        self.register_buffer(
            "_acc_count", torch.tensor([0.0], dtype=torch.float32)
        )
        self.register_buffer(
            "_num_accumulations", torch.tensor([0.0], dtype=torch.float32)
        )
        self.register_buffer(
            "_acc_sum", torch.zeros(num_features, dtype=torch.float32)
        )
        self.register_buffer(
            "_acc_sum_squared", torch.zeros(num_features, dtype=torch.float32)
        )

    def accumulate(self, attr: Tensor) -> None:
        if self._num_accumulations < self._max_accumulations:
            self._acc_count += attr.shape[0]
            self._num_accumulations += 1.0
            self._acc_sum += attr.sum(0)
            self._acc_sum_squared += attr.square().sum(0)

    def forward(self, attr: Tensor) -> Tensor:
        if self.training:
            self.accumulate(attr)

        return (attr - self.mean()) / self.std()

    def inverse(self, attr: Tensor) -> Tensor:
        return attr * self.std() + self.mean()

    def mean(self) -> Tensor:
        return self._acc_sum / self._acc_count.clamp_min(1.0)

    def std(self) -> Tensor:
        return torch.sqrt(
            self._acc_sum_squared / self._acc_count.clamp_min(1.0)
            - self.mean().square()
        ).clamp_min(self._std_epsilon)


class NaiveNormalizerBase:
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


class NaiveNodeNormalizer(NaiveNormalizerBase, NodeNormalizer):
    def __call__(
        self, node_key: str, attr_key: str, layout: TypedGraphLayout
    ) -> Tuple[NormalizerImpl, TypedGraphLayout]:
        return (
            NaiveNormalizerImpl(
                layout.node_sets[node_key].attrs[attr_key],
                self._max_accumulations,
                self._std_epsilon,
            ),
            layout,
        )


class NaiveEdgeNormalizer(NaiveNormalizerBase, EdgeNormalizer):
    def __call__(
        self, edge_key: str, layout: TypedGraphLayout
    ) -> Tuple[NormalizerImpl, TypedGraphLayout]:
        return (
            NaiveNormalizerImpl(
                layout.edge_sets[edge_key].attrs,
                self._max_accumulations,
                self._std_epsilon,
            ),
            layout,
        )


class NaiveLabelNormalizer(NaiveNormalizerBase, LabelNormalizer):
    def __call__(
        self, label_key: str, layout: TypedGraphLayout
    ) -> Tuple[NormalizerImpl, TypedGraphLayout]:
        return (
            NaiveNormalizerImpl(
                layout.labels[label_key].attrs,
                self._max_accumulations,
                self._std_epsilon,
            ),
            layout,
        )
