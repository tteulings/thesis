from typing import List

import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray
from scipy import spatial
from stl import mesh

import torch
from torch import Tensor

from models.bubble.data.config import SimulationConfig
from tggnn.data.index import SortedIndex
from tggnn.data.typed_graph import EdgeSet, NodeAttribute, NodeSet, TypedGraph
from tggnn.data.util import face_to_edge

from .remesh import remesh
from .replay import replay


def bubble_volume(
    positions: NDArray[np.float64], faces: NDArray[np.int32]
) -> Tensor:
    return (
        np.sum(
            positions[faces[:, 0]]
            * np.cross(
                positions[faces[:, 1]],
                positions[faces[:, 2]],
                axis=1,
            )
        )
    ) / 6.0


class Bubble(TypedGraph):
    def __init__(
        self,
        config: SimulationConfig,
        source_faces: NDArray[np.int32],
        source_connect: NDArray[np.int32],
        positions: List[NDArray[np.float64]],
        remesh_velocity: bool = False,
        target_acceleration: bool = False,
        old_velocity: bool = False,
    ):
        super().__init__()

        self._config = config
        self.old_velocity = old_velocity
        self.remesh_velocity = remesh_velocity
        self.target_acceleration = target_acceleration

        remesh_result = remesh(
            config,
            bubble_volume(positions[0], source_faces).item(),
            positions[1],
            source_faces,
            source_connect,
        )

        self.instructions = remesh_result.instructions

        if remesh_velocity:
            history = positions[1] - positions[0]
        else:
            history = positions[0]

        replay_result = replay(
            config,
            bubble_volume(positions[0], source_faces).item(),
            history,
            source_faces,
            source_connect,
            remesh_result.instructions,
            # remesh_velocity,
            True,
        )

        if remesh_velocity:
            velocities = replay_result.values
        else:
            velocities = remesh_result.positions - replay_result.values

        if target_acceleration:
            # a_2 = v_2 - v_1 = x_2 - 2x_1 + x_0
            target_tensor = (
                positions[2] - remesh_result.positions
            ) - velocities
        else:
            # v_2 = x_2 - x_1
            target_tensor = positions[2] - remesh_result.positions

        self.faces = torch.tensor(
            remesh_result.faces.transpose(),
            dtype=torch.long,
            device=self.device,
        )
        self.connect = torch.tensor(
            remesh_result.connect.transpose(),
            dtype=torch.int32,
            device=self.device,
        )
        self.positions = torch.tensor(
            remesh_result.positions, dtype=torch.float64, device=self.device
        )

        if self.old_velocity:
            velocities /= self._config.dt

        node_attr = torch.tensor(
            np.column_stack((velocities, la.norm(velocities, axis=1))),
            dtype=torch.float32,
        )

        self.add_node_set(
            "bubble", NodeSet({"velocity": NodeAttribute(node_attr)})
        )

        self.add_label_tensor(
            "target", torch.tensor(target_tensor, dtype=torch.float32)
        )

        # Mesh -> Graph
        mesh_index = SortedIndex(
            face_to_edge(self.faces, self.positions.size()[0])
        )

        float_positions = self.positions.float()

        # Compute mesh edge attributes.
        source, target = mesh_index.select_single(
            float_positions, float_positions
        )
        diffs = target - source

        mesh_edge_attr = torch.cat(
            (diffs, torch.norm(diffs, dim=1).unsqueeze(1)), 1
        )

        self.add_edge_set(
            "mesh",
            EdgeSet(mesh_index, mesh_edge_attr, "bubble", "bubble"),
        )

    def remesh(self, instructions: NDArray[np.ubyte]) -> "Bubble":
        cur_pos = self.positions.detach().cpu().numpy()

        velocity = self.node_sets["bubble"]["velocity"].attr[:, :3]

        if self.old_velocity:
            velocity *= self._config.dt

        values = (
            velocity.double()
            if self.remesh_velocity
            else (self.positions - velocity)
        )

        faces = np.ascontiguousarray(
            self.faces.detach().type(torch.int32).cpu().numpy().transpose()
        )
        connect = np.ascontiguousarray(
            self.connect.detach().cpu().numpy().transpose()
        )

        volume = self.volume().item()

        position_replay = replay(
            self._config, volume, cur_pos, faces, connect, instructions, False
        )
        value_replay = replay(
            self._config,
            volume,
            values.detach().cpu().numpy(),
            faces,
            connect,
            instructions,
            # self.remesh_velocity,
            True,
        )

        self.positions = torch.tensor(
            position_replay.values, device=self.device
        )

        self.faces = torch.tensor(
            position_replay.faces.transpose(),
            dtype=torch.long,
            device=self.device,
        )

        self.connect = torch.tensor(
            position_replay.connect.transpose(),
            dtype=torch.int32,
            device=self.device,
        )

        self.edge_sets["mesh"].index = SortedIndex(
            face_to_edge(self.faces, self.positions.size()[0])
        )

        new_vel = torch.tensor(
            value_replay.values
            if self.remesh_velocity
            else (position_replay.values - value_replay.values),
            dtype=torch.float32,
            device=self.device,
        )

        if self.old_velocity:
            new_vel /= self._config.dt

        self.node_sets["bubble"]["velocity"].attr = torch.cat(
            (new_vel, torch.norm(new_vel, dim=1, keepdim=True)), 1
        )

        float_positions = self.positions.float()

        source, target = self.edge_sets["mesh"].index.select_single(
            float_positions, float_positions
        )
        diffs = target - source

        self.edge_sets["mesh"].attr = torch.cat(
            (diffs, torch.norm(diffs, dim=1).unsqueeze(1)), 1
        )

        return self

    def update(self, delta: Tensor, do_remesh: bool = True) -> "Bubble":
        old_pos = self.positions.to(self.device)

        velocity = self.node_sets["bubble"]["velocity"].attr[:, :3].double()

        if self.old_velocity:
            velocity *= self._config.dt

        velocity = (
            (velocity + delta) if self.target_acceleration else delta.double()
        )
        new_pos = old_pos + velocity

        if do_remesh:
            history = velocity if self.remesh_velocity else old_pos

            volume = self.volume().item()

            faces = np.ascontiguousarray(
                self.faces.detach().type(torch.int32).cpu().numpy().transpose()
            )
            connect = np.ascontiguousarray(
                self.connect.detach().cpu().numpy().transpose()
            )

            result = remesh(
                self._config,
                volume,
                np.ascontiguousarray(new_pos.detach().cpu().numpy()),
                faces,
                connect,
            )

            replay_result = replay(
                self._config,
                volume,
                np.ascontiguousarray(history.detach().cpu().numpy()),
                faces,
                connect,
                result.instructions,
                # self.remesh_velocity,
                True,
            )

            self.faces = torch.tensor(
                result.faces.transpose(), dtype=torch.long, device=self.device
            )
            self.connect = torch.tensor(
                result.connect.transpose(),
                dtype=torch.int32,
                device=self.device,
            )

            new_pos = torch.tensor(result.positions, device=self.device)

            self.edge_sets["mesh"].index = SortedIndex(
                face_to_edge(self.faces, new_pos.size()[0])
            )

            velocity = (
                torch.tensor(
                    replay_result.values,
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.remesh_velocity
                else torch.tensor(
                    result.positions - replay_result.values,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        self.positions = new_pos

        if self.old_velocity:
            velocity /= self._config.dt

        self.node_sets["bubble"]["velocity"].attr = torch.cat(
            (velocity, torch.norm(velocity, dim=1).unsqueeze(1)), 1
        )

        float_positions = self.positions.float()

        source, target = self.edge_sets["mesh"].index.select_single(
            float_positions, float_positions
        )
        diffs = target - source

        self.edge_sets["mesh"].attr = torch.cat(
            (diffs, torch.norm(diffs, dim=1).unsqueeze(1)), 1
        )

        return self

        # Uniformly sample bubble nodes.
        # rng = np.random.default_rng(23)
        # selection = rng.choice(self.positions.shape[0], 300)

        # Find the vertically opposite nodes of the selection and construct the fluid edge index.
        # fluid_index = torch.from_numpy(
        #     np.stack([selection, find_nodes_below(self.positions, selection)], axis=0)
        # )
        # fluid_index = torch.cat(
        #     (
        #         fluid_index,
        #         fluid_index.index_select(0, torch.LongTensor([1, 0])),
        # ),
        # 1,
        # )

        # Compute the fluid edge attributes
        # position_pairs = self.positions[fluid_index.transpose(0, 1)]
        # diffs = position_pairs[:, 0] - position_pairs[:, 1]
        # fluid_edge_attr = torch.cat((diffs, torch.norm(diffs, dim=1).unsqueeze(1)), 1)

        # self.add_edge_set(
        #     "fluid", EdgeSet(fluid_index, fluid_edge_attr, "bubble", "bubble")
        # )

    def as_stl(self) -> mesh.Mesh:
        # Set up the mesh
        bubbleMesh = mesh.Mesh(
            np.zeros(self.faces.shape[1], dtype=mesh.Mesh.dtype)
        )

        # Set up the mesh position vectors
        bubbleMesh.vectors = (
            self.positions[self.faces.transpose(0, 1)].detach().cpu().numpy()
        )

        return bubbleMesh

    def volume(self) -> Tensor:
        return (
            self.positions[self.faces[0, :]]
            * self.positions[self.faces[1, :]].cross(
                self.positions[self.faces[2, :]], dim=1
            )
        ).sum() / 6.0

    def centroid(self) -> Tensor:
        center = self.positions[self.faces].sum(dim=0) / 4

        volume = (
            self.positions[self.faces[0, :]]
            * self.positions[self.faces[1, :]].cross(
                self.positions[self.faces[2, :]], dim=1
            )
        ).sum(dim=1)

        return volume.unsqueeze(1).mul(center).sum(dim=0) / volume.sum(dim=0)


def find_nodes_below(positions: Tensor, selection: np.ndarray):
    tree = spatial.KDTree(positions[:, :1])
    opposites = np.empty(selection.shape, dtype=selection.dtype)
    for i, node in enumerate(positions[selection]):
        closest = tree.query(node[:1], 10)[1][1:]
        opposites[i] = closest[
            (positions[closest, 2] - node[2]).abs().argmax().item()
        ]

    return opposites
