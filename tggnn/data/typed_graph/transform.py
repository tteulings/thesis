from typing import Generic, TypeVar

from ..abstract import DataTransform
from .typed_graph import TypedGraph


TG_data = TypeVar("TG_data", bound=TypedGraph)


class TypedGraphTransform(DataTransform[TG_data], Generic[TG_data]):
    pass
