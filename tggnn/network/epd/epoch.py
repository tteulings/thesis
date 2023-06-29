from typing import cast, Generic, Iterator, TypeVar, TYPE_CHECKING

from ...data.typed_graph import TypedGraph

if TYPE_CHECKING:
    from .epd import EncodeProcessDecode


TG_Data = TypeVar("TG_Data", bound=TypedGraph)


class EpochIterator(Generic[TG_Data], Iterator[int]):
    _cur: int
    _max: int
    _data: "EncodeProcessDecode[TG_Data]"

    def __init__(
        self, data: "EncodeProcessDecode[TG_Data]", num_epochs: int
    ) -> None:
        self._cur = 0
        self._max = num_epochs
        self._data = data

    def __iter__(self) -> "EpochIterator[TG_Data]":
        self._cur = 0
        return self

    def __next__(self) -> int:
        if self._cur < self._max:
            self._data.epoch += 1
            self._cur += 1
            return cast(int, self._data.epoch.item())

        raise StopIteration
