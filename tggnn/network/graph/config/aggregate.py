from torch.nn import Module

from ....common.util.aggregate import Aggregate


class AggregateConfig(Module):
    aggregate: Aggregate

    def __init__(
        self,
        aggregate: Aggregate,
    ) -> None:
        super().__init__()
        self.aggregate = aggregate

    def forward(self) -> None:
        raise NotImplementedError
