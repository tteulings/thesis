from copy import deepcopy
from typing import Dict, Optional, TypeVar

import torch
from torch import Tensor
from torch.types import Device

from .edge_set import EdgeSet, EdgeSetSummary
from .label import LabelSummary
from .node_set import NodeSet, NodeSetSummary

TG = TypeVar("TG", bound="TypedGraph")

# NOTE: Should TypedGraph inherit from 'object' such that users can add their
# own datastructures to the data? To do this correctly, pytorch functions such
# as '.to' should take the additional members into account. For an example
# implementation take a look at 'torch_geometric.data.Data'.
class TypedGraph:
    _node_sets: Dict[str, NodeSet]
    _edge_sets: Dict[str, EdgeSet]
    _labels: Dict[str, Tensor]
    device: Device
    base_layout: Optional["TypedGraphLayout"]

    def __init__(
        self,
        node_sets: Dict[str, NodeSet] = {},
        edge_sets: Dict[str, EdgeSet] = {},
        labels: Dict[str, Tensor] = {},
        device: Device = torch.device("cpu"),
        base_layout: Optional["TypedGraphLayout"] = None,
    ) -> None:
        self._node_sets = {}
        self._edge_sets = {}
        self._labels = {}

        self.base_layout = base_layout

        for key, node_set in node_sets.items():
            self.add_node_set(key, node_set)

        for key, edge_set in edge_sets.items():
            self.add_edge_set(key, edge_set)

        for key, label in labels.items():
            self.add_label_tensor(key, label)

        self.to(device)

    @property
    def node_sets(self) -> Dict[str, NodeSet]:
        return self._node_sets

    @property
    def edge_sets(self) -> Dict[str, EdgeSet]:
        return self._edge_sets

    @property
    def labels(self) -> Dict[str, Tensor]:
        return self._labels

    def clear(self) -> None:
        self._node_sets = {}
        self._edge_sets = {}
        self._labels = {}

    def merge_super_into(self, other: TG) -> TG:
        # Assure structural equivalence.
        if self.base_layout != other.layout():
            raise Exception(
                "Cannot merge TypedGraph instances, base layout is incompatible"
                " with the TypedGraph instance to merge."
            )

        # Update the base layout with correct attribute values.
        self.base_layout = other.layout()

        for key, edge_set in self._edge_sets.items():
            if edge_set.target in other.node_sets:
                other.node_sets[edge_set.target].edge_sets.append(key)

        for node_key in self._node_sets:
            if node_key in other.node_sets:
                node_set = self._node_sets[node_key]
                base_set = other.node_sets[node_key]

                # Add attributes not in other from self to other.
                for attr_key in (
                    node_set.attributes.keys() - base_set.attributes.keys()
                ):
                    base_set[attr_key] = node_set[attr_key]
            else:
                other.node_sets[node_key] = self._node_sets[node_key]

        other.edge_sets.update(self._edge_sets)
        other.labels.update(self._labels)

        self.clear()

        return other

    def extract_super_from(self: TG, other: "TypedGraph") -> TG:
        if self.base_layout is None:
            raise Exception(
                "Cannot extract super graph for TypedGraph without base_layout."
            )

        super_layout = other.layout()

        if not self.base_layout <= super_layout:
            raise Exception(
                "The base graph specified by `base_layout` is not contained"
                " within the supplied super graph. To merge a super graph into"
                " a supplied base graph, use `TypedGraph.merge_super_into`."
            )

        # Extract all node sets from other that are not in self.base_layout.
        for node_key, node_summary in super_layout.node_sets.items():
            if node_key not in self.base_layout.node_sets:
                self._node_sets[node_key] = other.node_sets[node_key]
                del other.node_sets[node_key]
            else:
                # Extract individual attributes.
                base_summary = self.base_layout.node_sets[node_key]

                super_keys = (
                    node_summary.attrs.keys() - base_summary.attrs.keys()
                )

                if len(super_keys) != 0:
                    node_set = NodeSet({}, base_summary)

                    for attr_key in super_keys:
                        node_set[attr_key] = other.node_sets[node_key][attr_key]
                        del other.node_sets[node_key][attr_key]

                    node_set.edge_sets = [
                        edge_key
                        for edge_key in other.node_sets[node_key].edge_sets
                        if edge_key not in self.base_layout.edge_sets
                    ]

                    self._node_sets[node_key] = node_set

        # Extract all edge sets from other that are not in self.base_layout.
        for edge_key, summary in super_layout.edge_sets.items():
            if edge_key not in self.base_layout.edge_sets:
                self._edge_sets[edge_key] = other.edge_sets[edge_key]

                if summary.target in self.base_layout.node_sets:
                    other.node_sets[summary.target].edge_sets.remove(edge_key)

                del other.edge_sets[edge_key]

        # Extract all labels from other that are not in self.base_layout.
        for label_key in super_layout.labels:
            if label_key not in self.base_layout.labels:
                self._labels[label_key] = other.labels[label_key]
                del other.labels[label_key]

        # Update the base layout with correct attribute values.
        self.base_layout = other.layout()

        return self

    def add_node_set(self, name: str, node_set: NodeSet) -> None:
        if self.base_layout is not None:
            if name in self.base_layout.node_sets:
                for attr_name in self.base_layout.node_sets[name].attrs:
                    if attr_name in node_set.attributes:
                        raise Exception(
                            f'A NodeAttribute with name "{attr_name}" is'
                            f' already present on the "{name}" NodeSet in the'
                            " base layout of this TypedGraph."
                        )

                # Inherit the base layout of this typed graph.
                node_set.base_layout = self.base_layout.node_sets[name]

        if name in self._node_sets:
            raise Exception(
                f'A NodeSet with name "{name}" is already present in this'
                " TypedGraph."
            )

        self._node_sets[name] = node_set

    def add_edge_set(self, name: str, edge_set: EdgeSet) -> None:
        if self.base_layout is not None:
            if name in self.base_layout.edge_sets:
                raise Exception(
                    f'An EdgeSet with name "{name}" is already present in the'
                    " base layout of this TypedGraph."
                )

        if name in self._edge_sets:
            raise Exception(
                f'An EdgeSet with name "{name}" is already present in this'
                " TypedGraph."
            )

        if not (
            edge_set.source in self._node_sets
            or (
                False
                if self.base_layout is None
                else edge_set.source in self.base_layout.node_sets
            )
        ):
            raise Exception(
                "This TypedGraph or its base layout does not contain the"
                f' required source set ("{edge_set.source}") for this edge '
                f'set: "{name}".'
            )

        if edge_set.target in self._node_sets:
            # Add this edge set to the edge set list of the target node set.
            self._node_sets[edge_set.target].edge_sets.append(name)
        elif not (
            False
            if self.base_layout is None
            else edge_set.target in self.base_layout.node_sets
        ):
            raise Exception(
                "This TypedGraph or its base layout does not contain the"
                f' required target set ("{edge_set.target}") for this edge '
                f'set: "{name}".'
            )

        # Add the edge set.
        self._edge_sets[name] = edge_set

    def add_label_tensor(self, name: str, tensor: Tensor) -> None:
        if name in self._edge_sets:
            raise Exception(
                f'A Label with name "{name}" is already present in this'
                " TypedGraph."
            )

        self._labels[name] = tensor

    def to(self, device: Device) -> None:
        for node_set in self._node_sets.values():
            node_set.to(device)

        for edge_set in self._edge_sets.values():
            edge_set.to(device)

        for key, label in self._labels.items():
            self._labels[key] = label.to(device)

        self.device = device

    def layout(self, include_base: bool = False) -> "TypedGraphLayout":
        layout = TypedGraphLayout(self)

        if include_base and self.base_layout is not None:
            # NOTE: Use deepcopy to make sure we don't carry references to the
            # base layout.
            layout.node_sets.update(deepcopy(self.base_layout.node_sets))

            for key, edge_set in layout.edge_sets.items():
                if edge_set.target in self.base_layout.node_sets:
                    layout.node_sets[edge_set.target].edge_sets.append(key)

            layout.edge_sets.update(deepcopy(self.base_layout.edge_sets))
            layout.labels.update(deepcopy(self.base_layout.labels))

        return layout


class TypedGraphLayout:
    node_sets: Dict[str, NodeSetSummary]
    edge_sets: Dict[str, EdgeSetSummary]
    labels: Dict[str, LabelSummary]

    def __init__(self, graph: TypedGraph) -> None:

        self.node_sets = {
            key: node_set.summarize()
            for key, node_set in graph.node_sets.items()
        }
        self.edge_sets = {
            key: edge_set.summarize()
            for key, edge_set in graph.edge_sets.items()
        }
        self.labels = {
            key: LabelSummary(label) for key, label in graph.labels.items()
        }

    def __le__(self, other: "TypedGraphLayout") -> bool:
        return (
            all(
                key in other.node_sets and node_set <= other.node_sets[key]
                for key, node_set in self.node_sets.items()
            )
            and all(
                key in other.edge_sets and edge_set == other.edge_sets[key]
                for key, edge_set in self.edge_sets.items()
            )
            and all(key in other.labels for key in self.labels)
        )

    def __eq__(self, other: "TypedGraphLayout") -> bool:
        return (
            self.node_sets == other.node_sets
            and self.edge_sets == other.edge_sets
            and self.labels == other.labels
        )

    def __repr__(self) -> str:
        return str(self.__dict__)
