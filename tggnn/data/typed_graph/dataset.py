import abc
from typing import cast, Generic, List, Optional, TypeVar

import os
import jsonpickle

from torch.utils.data import Dataset

from .typed_graph import TypedGraph, TypedGraphLayout
from .transform import TypedGraphTransform


TG_Data = TypeVar("TG_Data", bound=TypedGraph)


class TypedGraphDataset(
    Dataset[TG_Data], Generic[TG_Data], metaclass=abc.ABCMeta
):
    root: str
    transforms: Optional[List[TypedGraphTransform[TG_Data]]]
    _layout: TypedGraphLayout

    def __init__(
        self,
        root: str,
        transforms: Optional[List[TypedGraphTransform[TG_Data]]],
    ) -> None:
        super().__init__()

        self.root = root
        self.transforms = transforms

        with open(os.path.join(root, "layout.json"), "r") as file:
            self._layout = cast(
                TypedGraphLayout, jsonpickle.decode(file.read())
            )

    def layout(self) -> TypedGraphLayout:
        return self._layout

    def __getitem__(self, index: int) -> TG_Data:
        data = self.__get__(index)

        if self.transforms is not None:
            for transform in self.transforms:
                data = transform(data)

        return data

    @abc.abstractmethod
    def __get__(self, idx: int) -> TG_Data:
        pass

    # NOTE: Technically, this is unnecessary as it is already defined in the
    # Dataset base class. However, I like to keep it here for clarity.
    @abc.abstractmethod
    def __len__() -> int:
        pass
