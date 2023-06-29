from typing import Dict, Generator, List, Tuple, TypeVar

from ....data.typed_graph import TypedGraphLayout
from ...graph.config import (
    AggregateConfig,
    GraphBlockConfig,
    GraphPassConfig,
    TransferConfig,
    UpdateConfig,
)
from ...modules.base_types import (
    EdgeDecoder,
    NodeDecoder,
    EdgeEncoder,
    NodeEncoder,
    EdgeNormalizer,
    NodeNormalizer,
    LabelNormalizer,
)
from .block import BlockConfig
from .edge import EdgeFunctionSet
from .label import LabelFunctionSet
from .node import NodeFunctionSet


T_Module = TypeVar("T_Module")
TupleGenerator = Generator[Tuple[str, T_Module], None, None]
NestedGenerator = Generator[Tuple[str, TupleGenerator[T_Module]], None, None]


class EPDConfig:
    _blocks: List[BlockConfig]
    _nodes: Dict[str, Dict[str, NodeFunctionSet]]
    _edges: Dict[str, EdgeFunctionSet]
    _labels: Dict[str, LabelFunctionSet]

    def __init__(
        self,
        blocks: List[BlockConfig],
        nodes: Dict[str, Dict[str, NodeFunctionSet]],
        edges: Dict[str, EdgeFunctionSet],
        labels: Dict[str, LabelFunctionSet],
    ) -> None:
        self._blocks = blocks
        self._nodes = nodes
        self._edges = edges
        self._labels = labels

    def split_config(
        self, layout: TypedGraphLayout
    ) -> Tuple[
        NestedGenerator[NodeNormalizer],
        TupleGenerator[EdgeNormalizer],
        TupleGenerator[LabelNormalizer],
        NestedGenerator[NodeEncoder],
        TupleGenerator[EdgeEncoder],
        NestedGenerator[NodeDecoder],
        TupleGenerator[EdgeDecoder],
        List[GraphBlockConfig],
    ]:

        node_normalizers = (
            (
                key,
                (
                    (attr, function_set.normalizer)
                    for attr, function_set in node.items()
                    if function_set.normalizer is not None
                ),
            )
            for key, node in self._nodes.items()
        )
        # node_normalizers = (
        #     (key, function_set.normalizer)
        #     for key, function_set in self._nodes.items()
        #     if function_set.normalizer is not None
        # )
        edge_normalizers = (
            (key, function_set.normalizer)
            for key, function_set in self._edges.items()
            if function_set.normalizer is not None
        )
        label_normalizers = (
            (key, function_set.normalizer)
            for key, function_set in self._labels.items()
            if function_set.normalizer is not None
        )

        node_encoders = (
            (
                key,
                (
                    (attr, function_set.encoder)
                    for attr, function_set in node.items()
                ),
            )
            for key, node in self._nodes.items()
        )
        edge_encoders = (
            (key, function_set.encoder)
            for key, function_set in self._edges.items()
            if function_set.encoder is not None
        )

        node_decoders = (
            (
                key,
                (
                    (attr, function_set.decoder)
                    for attr, function_set in node.items()
                    if function_set.decoder is not None
                ),
            )
            for key, node in self._nodes.items()
        )
        edge_decoders = (
            (key, function_set.decoder)
            for key, function_set in self._edges.items()
            if function_set.decoder is not None
        )

        block_configs = [
            GraphBlockConfig(
                GraphPassConfig(
                    updates={
                        key: {
                            attr: UpdateConfig(
                                self._nodes[key][attr].update,
                                self._nodes[key][attr].residual,
                            )
                            for attr in layout.node_sets[key].attrs
                        }
                        for key in node_keys
                    },
                    transfers={
                        key: TransferConfig(
                            self._edges[key].transfer, self._edges[key].residual
                        )
                        for key in edge_keys
                    },
                    aggregates={
                        key: AggregateConfig(self._edges[key].aggregate)
                        for key in edge_keys
                    },
                ),
                iterations,
            )
            for node_keys, edge_keys, iterations in map(
                lambda block: (
                    block.node_sets,
                    [
                        key
                        for node in block.node_sets
                        for key in layout.node_sets[node].edge_sets
                    ],
                    block.iterations,
                ),
                self._blocks,
            )
        ]

        return (
            node_normalizers,
            edge_normalizers,
            label_normalizers,
            node_encoders,
            edge_encoders,
            node_decoders,
            edge_decoders,
            block_configs,
        )
