from tggnn.common.util.aggregate import Aggregate
from tggnn.data.typed_graph import TypedGraphLayout
from tggnn.model import BasicModel
from tggnn.network.epd import (
    BlockConfig,
    EdgeFunctionSet,
    EPDConfig,
    LabelFunctionSet,
    NodeFunctionSet,
)
from tggnn.network.modules.decoder import DenseNodeDecoder
from tggnn.network.modules.encoder import DenseEdgeEncoder, DenseNodeEncoder
from tggnn.network.modules.normalizer import (
    WelfordEdgeNormalizer,
    WelfordLabelNormalizer,
    WelfordNodeNormalizer,
)
from tggnn.network.modules.transfer import DenseTransfer
from tggnn.network.modules.update import DenseUpdate

from ..data import Bubble


def bubble_basic_model(
    data_layout: TypedGraphLayout,
    latent_size: int,
    num_layers: int,
    iterations: int,
) -> BasicModel[Bubble]:
    # Setup the node update and edge transfer functions.
    hiddens = num_layers * [latent_size]
    decode_hiddens = (num_layers - 1 if num_layers > 0 else 0) * [latent_size]

    config = EPDConfig(
        blocks=[
            BlockConfig(["bubble"], iterations),
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
        },
        labels={
            "target": LabelFunctionSet(
                normalizer=WelfordLabelNormalizer(),
            )
        },
    )

    return BasicModel[Bubble](config, data_layout)
