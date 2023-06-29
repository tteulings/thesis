from typing import Generic, TypeVar
import abc

T_Data = TypeVar("T_Data")


class DataTransform(Generic[T_Data], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, data: T_Data) -> T_Data:
        pass
