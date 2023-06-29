import torch
from models.flag.data.util import NodeType
from tggnn.data.typed_graph import TypedGraphTransform

from .flag import Flag


class AdditiveNoiseTransform(TypedGraphTransform[Flag]):
    def __init__(self, std: float, gamma: float = 0.1) -> None:
        super().__init__()

        self.std = std
        self.gamma = gamma

    def __call__(self, data: Flag) -> Flag:
        device = data.device

        mask = (data.node_type == NodeType.NORMAL).to(device)

        noise = torch.normal(
            mean=0, std=self.std, size=data.positions.size(), device=device
        )

        noise = noise.where(mask, torch.zeros_like(noise))

        data.positions = data.positions.to(device) + noise
        data.node_sets["flag"]["velocity"].attr[:, :3] += noise
        data.node_sets["flag"]["velocity"].attr[:, 3] = (
            data.node_sets["flag"]["velocity"].attr[:, :3].norm(dim=1)
        )

        float_positions = data.positions.float()

        source, target = data.edge_sets["mesh"].index.select_single(
            float_positions, float_positions
        )

        diffs = target - source

        data.edge_sets["mesh"].attr = torch.cat(
            (diffs, torch.norm(diffs, dim=1, keepdim=True)), 1
        )

        data.labels["target"] -= (1 + self.gamma) * noise

        return data
