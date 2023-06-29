import abc
from typing import Dict, List, Optional, Tuple

from torch import Tensor
from torch.nn import Module

from ..base import SetModule


class TransferModuleImpl(Module, metaclass=abc.ABCMeta):
    def __call__(
        self,
        source: Dict[str, Tensor],
        target: Dict[str, Tensor],
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        return self.forward(source, target, edge_attr)

    @abc.abstractmethod
    def forward(
        self,
        source: Dict[str, Tensor],
        target: Dict[str, Tensor],
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        pass

    @abc.abstractmethod
    def attribute_keys(self) -> Tuple[List[str], List[str]]:
        pass


TransferModule = SetModule[TransferModuleImpl]
