from typing import Tuple

import torch

from tggnn.data.index import AllToOneIndex, OneToAllIndex
from tggnn.data.state import GlobalState
from tggnn.data.typed_graph import (
    EdgeSet,
    NodeAttribute,
    NodeSet,
    TypedGraph,
    TypedGraphLayout,
)

from tggnn.common.util.aggregate import Aggregate
from tggnn.network.epd import (
    BlockConfig,
    EdgeFunctionSet,
    EPDConfig,
    LabelFunctionSet,
    NodeFunctionSet,
)

from tggnn.network.modules.decoder import DenseNodeDecoder, DenseCentroidDecoder
from tggnn.network.modules.encoder import (
    DenseNodeEncoder,
    DenseEdgeEncoder,
    NoOpNodeEncoder,
    DenseCentroidEncoder
)
from tggnn.network.modules.normalizer import (
    WelfordEdgeNormalizer,
    WelfordLabelNormalizer,
    WelfordNodeNormalizer,
)
from tggnn.network.modules.transfer import DenseTransfer
from tggnn.network.modules.update import DenseUpdate, MemoryUpdate, DenseCentroidUpdate

from tggnn.model import StatefulModel

from ..data import Bubble


def bubble_memory_model(
    data_layout: TypedGraphLayout,
    latent_size: int,
    num_layers: int,
    iterations: int,
) -> Tuple[StatefulModel[Bubble, TypedGraph], TypedGraph]:
    

    print(iterations)
    # NOTE: The feature tensors of "save" and "load" are initialized to size
    # (0, 4), to provide the system with the information that both types
    # initially have 4 attributes (after calling "center_memory").
    state = GlobalState(
        data_layout=data_layout,
        node_sets={
            "centroid": NodeSet(
                {"memory": NodeAttribute(torch.zeros((1, latent_size))), "velocity": NodeAttribute(torch.zeros((1,3)))},
                
            )
        },
        edge_sets={
            "save": EdgeSet(
                AllToOneIndex(), torch.empty((0, 4)), "bubble", "centroid"
            ),
            "load": EdgeSet(
                OneToAllIndex(), torch.empty((0, 4)), "centroid", "bubble"
            ),
        },
    )

    # Setup the node update and edge transfer functions.
    hiddens = num_layers * [latent_size]
    decode_hiddens = (num_layers - 1 if num_layers > 0 else 0) * [latent_size]

    # FIXME: Should we use ReLU on last layers (activate_final)?
    config = EPDConfig(
        blocks=[
            BlockConfig(["bubble"], iterations),
            BlockConfig(["centroid"], 1),
        ],
        nodes={
            "bubble": {
                "velocity": NodeFunctionSet(
                    update=DenseUpdate(hiddens),
                    encoder=DenseNodeEncoder(hiddens),
                    decoder=DenseNodeDecoder(decode_hiddens, "target"),
                    normalizer=WelfordNodeNormalizer(),
                    residual=True,
                )
            },
            "centroid": {
                "memory": NodeFunctionSet(
                    update=MemoryUpdate(),
                    encoder=NoOpNodeEncoder(),
                    normalizer=None,
                ),

                'velocity': NodeFunctionSet(
                    encoder=DenseCentroidEncoder(hiddens),

                    update=DenseCentroidUpdate(hiddens),

                    decoder=DenseCentroidDecoder(decode_hiddens, "target"),
                    normalizer=None,


                )

            },
        },
        edges={
            "mesh": EdgeFunctionSet(
                transfer=DenseTransfer(
                    hiddens,
                    source_attrs=["velocity"],
                    target_attrs=["velocity"],
                ),
                encoder=DenseEdgeEncoder(hiddens),
                normalizer=WelfordEdgeNormalizer(),
                aggregate=Aggregate.SUM,
                residual=True,
            ),
            "save": EdgeFunctionSet(
                transfer=DenseTransfer(
                    hiddens, source_attrs=["velocity"], target_attrs=["memory"]
                ),
                encoder=DenseEdgeEncoder(hiddens),
                normalizer=WelfordEdgeNormalizer(),
                aggregate=Aggregate.MEAN,
            ),
            "load": EdgeFunctionSet(
                transfer=DenseTransfer(
                    hiddens, source_attrs=["memory"], target_attrs=["velocity"]
                ),
                encoder=DenseEdgeEncoder(hiddens),
                normalizer=WelfordEdgeNormalizer(),
                aggregate=Aggregate.MEAN,
            ),
        },
        labels={
            "target": LabelFunctionSet(
                normalizer=WelfordLabelNormalizer(),
            )
        },
    )

    return (
        StatefulModel[Bubble, TypedGraph](config, state.layout(True)),
        state,
    )


# TODO: We could add pre- and post-hooks to StatefulModel that execute before
# and after the forward call (signature:
# Callable[[TG_Data, TG_State], Tuple[TG_Data, TG_State]]). This could then be
# registered as a pre-hook.
# FIXME: Should positions just be a node attribute of the mesh nodes?
def center_memory(data: Bubble, state: TypedGraph) -> TypedGraph:
    data.positions = data.positions.to(state.device)

    positions = data.positions.float()
    centroid = data.centroid().float().unsqueeze_(0)

    node_positions, center_vec = state.edge_sets["save"].index.select_single(
        positions, centroid
    )

    memory_pos_diff = center_vec - node_positions

    state.edge_sets["save"].attr = torch.cat(
        (memory_pos_diff, memory_pos_diff.norm(dim=1).unsqueeze(1)),
        1,
    )

    memory_pos_diff.neg_()

    state.edge_sets["load"].attr = torch.cat(
        (memory_pos_diff, memory_pos_diff.norm(dim=1).unsqueeze(1)),
        1,
    )

    return state
