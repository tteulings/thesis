import abc
from typing import Generic, Tuple, TypeVar

from torch.nn import Module

from ..data.typed_graph import TypedGraph
from ..network.epd import EncodeProcessDecode

TG_Data = TypeVar("TG_Data", bound=TypedGraph)
TG_State = TypeVar("TG_State", bound=TypedGraph)


class StatefulModelBase(
    Generic[TG_Data, TG_State], Module, metaclass=abc.ABCMeta
):
    @abc.abstractmethod
    def forward(
        self, data: TG_Data, global_state: TG_State
    ) -> Tuple[TG_Data, TG_State]:
        pass


class StatefulModel(EncodeProcessDecode[TG_Data], Generic[TG_Data, TG_State]):
    def forward(
        self, data: TG_Data, global_state: TG_State
    ) -> Tuple[TG_Data, TG_State]:
        graph = global_state.merge_super_into(data)

        graph, center_out = super().forward(graph)
        # print(graph)

        global_state = global_state.extract_super_from(graph)

        return (graph, global_state, center_out)

    def gather_statistics(
        self, data: TG_Data, global_state: TG_State
    ) -> Tuple[TG_Data, TG_State]:
        graph = global_state.merge_super_into(data)

        super().gather_statistics(graph)

        global_state = global_state.extract_super_from(graph)

        return (data, global_state)
