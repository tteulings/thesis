import abc
from typing import cast

from os import path

import jsonpickle

from .typed_graph import TypedGraphLayout


class TypedGraphProcessor(metaclass=abc.ABCMeta):
    root: str

    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = root

    def process(self, write_layout: bool = True) -> None:
        layout = self.__process__()

        if write_layout:
            with open(path.join(self.root, "layout.json"), "w") as file:
                file.write(cast(str, jsonpickle.encode(layout)))

    @abc.abstractmethod
    def __process__(self) -> TypedGraphLayout:
        pass
