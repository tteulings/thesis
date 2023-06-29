import abc

from torch import Tensor
from torch.nn import Module

from ..base import AttributeModule, SetModule


class NormalizerImpl(Module, metaclass=abc.ABCMeta):
    def __call__(self, attr: Tensor) -> Tensor:
        return self.forward(attr)

    @abc.abstractmethod
    def forward(self, attr: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def accumulate(self, attr: Tensor) -> None:
        pass

    @abc.abstractmethod
    def inverse(self, attr: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def mean(self) -> Tensor:
        pass

    @abc.abstractmethod
    def std(self) -> Tensor:
        pass


# T_Normalizer = TypeVar("T_Normalizer", bound=NormalizerType)

# NOTE: Static methods on generic parameters cannot simply be called as you
# might expect (e.g. `T.static_method()`). However, we can work around this by
# retrieving the generic class instantiation by accessing
# `self.__orig_class__.__args__[0]`.
# source: <https://stackoverflow.com/a/67740050/11665573>
# class Normalizer(
#     Generic[T_Normalizer], LazyModule[NormalizerImpl], metaclass=abc.ABCMeta
# ):
#     def type(self) -> T_Normalizer:
#         return self.__orig_class__.__args__[0]  # type: ignore

NodeNormalizer = AttributeModule[NormalizerImpl]
EdgeNormalizer = SetModule[NormalizerImpl]
LabelNormalizer = SetModule[NormalizerImpl]
