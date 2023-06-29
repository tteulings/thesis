from dataclasses import dataclass

from .graph_pass import GraphPassConfig


@dataclass
class GraphBlockConfig:
    graph_config: GraphPassConfig
    iterations: int
