from dataclasses import dataclass
from typing import List


@dataclass
class BlockConfig:
    node_sets: List[str]
    # edge_sets: List[str],
    iterations: int
