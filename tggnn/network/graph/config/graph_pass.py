from dataclasses import dataclass
from typing import Dict

from .aggregate import AggregateConfig
from .transfer import TransferConfig
from .update import UpdateConfig


@dataclass
class GraphPassConfig:
    updates: Dict[str, Dict[str, UpdateConfig]]
    transfers: Dict[str, TransferConfig]
    aggregates: Dict[str, AggregateConfig]
