from typing import Generic, TypeVar


from ..data.typed_graph import TypedGraph
from ..network.epd import EncodeProcessDecode

TG_Data = TypeVar("TG_Data", bound=TypedGraph)


class BasicModel(EncodeProcessDecode[TG_Data], Generic[TG_Data]):
    pass
