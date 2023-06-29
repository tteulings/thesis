from typing import Optional

from torch import Tensor
from torch.types import Device

from ..abstract import Index


class EdgeSet:
    index: Index
    attr: Optional[Tensor]
    source: str
    target: str

    def __init__(
        self,
        index: Index,
        features: Optional[Tensor],
        source: str,
        target: str,
    ) -> None:
        self.index = index
        self.attr = features
        self.source = source
        self.target = target

    def num_attr(self) -> int:
        if self.attr is None:
            return 0

        return self.attr.size()[1]

    def to(self, device: Device) -> None:
        self.index.to(device)

        if self.attr is not None:
            self.attr = self.attr.to(device)

    def summarize(self) -> "EdgeSetSummary":
        return EdgeSetSummary(self)


class EdgeSetSummary:
    attrs: int
    source: str
    target: str

    def __init__(self, edge_set: EdgeSet) -> None:
        self.attrs = edge_set.num_attr()
        self.source = edge_set.source
        self.target = edge_set.target

    def __eq__(self, other: "EdgeSetSummary") -> bool:
        return self.source == other.source and self.target == other.target

    def __repr__(self) -> str:
        return str(self.__dict__)
