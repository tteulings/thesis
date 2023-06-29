from typing import Dict

import torch
from torch import Tensor
from torch.types import Device

from .typed_graph import EdgeSet, NodeSet, TypedGraph, TypedGraphLayout


def GlobalState(
    data_layout: TypedGraphLayout,
    node_sets: Dict[str, NodeSet] = {},
    edge_sets: Dict[str, EdgeSet] = {},
    labels: Dict[str, Tensor] = {},
    device: Device = torch.device("cpu"),
) -> TypedGraph:
    return TypedGraph(node_sets, edge_sets, labels, device, data_layout)
