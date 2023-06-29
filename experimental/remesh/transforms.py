import torch

from tggnn.data.typed_graph import TypedGraphTransform

from .bubble import Bubble


class AdditiveNoiseTransform(TypedGraphTransform[Bubble]):
    def __init__(self, std: float, gamma: float = 0.1) -> None:
        super().__init__()

        self.std = std
        self.gamma = gamma

    def __call__(self, data: Bubble) -> Bubble:
        device = data.device

        noise = torch.normal(
            mean=0, std=self.std, size=data.positions.size()
        ).to(device)

        data.positions = data.positions.to(device) + noise
        data.node_sets["bubble"]["velocity"].attr[:, :3] += noise
        data.node_sets["bubble"]["velocity"].attr[:, 3] = (
            data.node_sets["bubble"]["velocity"].attr[:, :3].norm(dim=1)
        )

        float_positions = data.positions.float()

        source, target = data.edge_sets["mesh"].index.select_single(
            float_positions, float_positions
        )

        diffs = target - source

        data.edge_sets["mesh"].attr = torch.cat(
            (diffs, torch.norm(diffs, dim=1, keepdim=True)), 1
        )

        if data.target_acceleration:
            data.labels["target"] -= (1 + self.gamma) * noise
        else:
            data.labels["target"] -= noise

        return data


class TimeShiftTransform(TypedGraphTransform[Bubble]):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: Bubble) -> Bubble:
        shift = data._config.dt

        vel = data.node_sets["bubble"]["velocity"]

        vel.attr[:, :3] /= shift
        vel.attr[:, 3] = vel.attr[:, :3].norm(dim=1)

        return data


class RelativeNoiseTransform(TypedGraphTransform[Bubble]):
    def __init__(self, std: float, gamma: float = 0.1) -> None:
        super().__init__()

        self.std = std
        self.gamma = gamma

    def __call__(self, data: Bubble) -> Bubble:
        # mesh_index = data.edge_sets["mesh"].index.transpose(0, 1)
        # mesh_index = data.edge_sets["mesh"].index.select()

        device = data.device

        # noise = torch.normal(mean=0.0, std=self.std, size=pos.size()).to(device)
        # new_pos = data.positions.to(device) + noise

        distr = torch.normal(
            mean=0, std=self.std, size=data.positions.size()
        ).to(device)

        noise = distr * data.node_sets["bubble"]["velocity"].attr[:, :3]

        data.positions = data.positions.to(device) + noise
        data.node_sets["bubble"]["velocity"].attr[:, :3] += noise
        data.node_sets["bubble"]["velocity"].attr[:, 3] = (
            data.node_sets["bubble"]["velocity"].attr[:, :3].norm(dim=1)
        )

        float_positions = data.positions.float()

        source, target = data.edge_sets["mesh"].index.select_single(
            float_positions, float_positions
        )

        diffs = target - source

        data.edge_sets["mesh"].attr = torch.cat(
            (diffs, torch.norm(diffs, dim=1, keepdim=True)), 1
        )

        if data.target_acceleration:
            data.labels["target"] -= (1 + self.gamma) * noise
        else:
            data.labels["target"] -= noise

        return data

        # new_vel = data.labels["target"] * noise

        # new_pos = data.positions.to(device) - (new_vel - data.labels["target"])

        # source, target = data.edge_sets["mesh"].index.select_single(
        #     new_pos, new_pos
        # )
        # diffs = target - source

        # # NOTE: This method of diff computation might seem strange, but I
        # # believe it will better respect machine precision.
        # # pos = data.positions.to(device)
        # # vel_diff = new_vel - data.labels["bubble"]

        # # source, target = data.edge_sets["mesh"].index.select(pos, pos)
        # # pos_diffs = target - source
        # # vel_diffs = vel_diff[mesh_index[:, 0]] - vel_diff[mesh_index[:, 1]]

        # # diffs = pos_diffs + vel_diffs

        # data.edge_sets["mesh"].attr = torch.cat(
        #     (diffs, torch.norm(diffs, dim=1).unsqueeze(1)), 1
        # )

        # # data.labels["bubble"] -= noise
        # data.positions = new_pos
        # data.labels["target"] = new_vel

        # return data
