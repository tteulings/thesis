import abc
from typing import Dict, List, Tuple, TYPE_CHECKING

from torch import Tensor
from torch.types import Device

from ...common.util.aggregate import Aggregate

if TYPE_CHECKING:
    from ..typed_graph import NodeSet


class Index(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select(
        self,
        source: "NodeSet",
        source_keys: List[str],
        target: "NodeSet",
        target_keys: List[str],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        pass

    @abc.abstractmethod
    def select_single(
        self, source: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        pass

    @abc.abstractmethod
    def gather(self, edge_attr: Tensor, aggr: Aggregate) -> Tensor:
        pass

    @abc.abstractmethod
    def to(self, device: Device) -> None:
        pass
