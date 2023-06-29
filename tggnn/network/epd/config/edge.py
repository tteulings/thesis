from dataclasses import dataclass
from typing import Optional

from ....common.util.aggregate import Aggregate
from ...modules.base_types import (
    EdgeDecoder,
    EdgeEncoder,
    EdgeNormalizer,
    TransferModule,
)


@dataclass
class EdgeFunctionSet:
    transfer: TransferModule
    aggregate: Aggregate
    encoder: Optional[EdgeEncoder] = None
    decoder: Optional[EdgeDecoder] = None
    normalizer: Optional[EdgeNormalizer] = None
    residual: bool = False
