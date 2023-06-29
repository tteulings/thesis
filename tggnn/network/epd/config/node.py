from dataclasses import dataclass
from typing import Optional

from ...modules.base_types import (
    NodeDecoder,
    NodeEncoder,
    NodeNormalizer,
    UpdateModule,
)


@dataclass
class NodeFunctionSet:
    update: UpdateModule
    encoder: NodeEncoder
    decoder: Optional[NodeDecoder] = None
    normalizer: Optional[NodeNormalizer] = None
    residual: bool = False
