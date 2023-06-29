from typing import Dict, List, Optional

from torch import Tensor
from torch.types import Device


class NodeAttribute:
    attr: Tensor

    def __init__(self, attr: Tensor) -> None:
        self.attr = attr

    def num_attr(self) -> int:
        return self.attr.size()[1]

    def num_nodes(self) -> int:
        return self.attr.size()[0]

    def to(self, device: Device) -> None:
        self.attr = self.attr.to(device)


class NodeSet:
    _attributes: Dict[str, NodeAttribute]

    base_layout: Optional["NodeSetSummary"]
    edge_sets: List[str]

    def __init__(
        self,
        attributes: Dict[str, NodeAttribute],
        base_layout: "NodeSetSummary" = None,
    ) -> None:
        self._attributes = attributes

        self.base_layout = base_layout
        self.edge_sets = []

    def __getitem__(self, key: str) -> NodeAttribute:
        return self._attributes[key]

    def __setitem__(self, name: str, attribute: NodeAttribute) -> None:
        if self.base_layout is not None:
            if name in self.base_layout.attrs:
                raise Exception(
                    f'A NodeAttribute with name "{name}" is already present in'
                    " the base layout of this NodeSet."
                )

        if name in self._attributes:
            raise Exception(
                f'A NodeAttribute with name "{name}" is already present in this'
                " NodeSet."
            )

        self._attributes[name] = attribute

    def __delitem__(self, name: str) -> None:
        del self._attributes[name]

    def add_attribute(self, key: str, attribute: NodeAttribute) -> None:
        self.__setitem__(key, attribute)

    @property
    def attributes(self) -> Dict[str, NodeAttribute]:
        return self._attributes

    def to(self, device: Device) -> None:
        for attribute in self._attributes.values():
            attribute.to(device)

    def summarize(self) -> "NodeSetSummary":
        return NodeSetSummary(self)


class NodeSetSummary:
    attrs: Dict[str, int]
    edge_sets: List[str]

    def __init__(self, node_set: NodeSet) -> None:
        self.attrs = {
            key: attribute.num_attr()
            for key, attribute in node_set.attributes.items()
        }
        self.edge_sets = node_set.edge_sets.copy()

    def __le__(self, other: "NodeSetSummary") -> bool:
        return all(key in other.edge_sets for key in self.edge_sets) and all(
            key in other.attrs for key in self.attrs
        )

    def __eq__(self, other: "NodeSetSummary") -> bool:
        return (
            self.edge_sets == other.edge_sets
            and self.attrs.keys() == other.attrs.keys()
        )

    def __repr__(self) -> str:
        return str(self.__dict__)
