from numpy.typing import NDArray
from typing import List

import numpy as np
from numpy import linalg as la
from scipy import spatial
from stl import mesh

import torch
from torch import Tensor

from tggnn.data.index import SortedIndex
from tggnn.data.typed_graph import EdgeSet, NodeAttribute, NodeSet, TypedGraph
from tggnn.data.util import face_to_edge

from .bin import BinFile
from .config import SimulationConfig
from .remesh import remesh


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
    ):
        super().__init__()

        self._config = config
        self.remesh_velocity = remesh_velocity
        self.target_acceleration = target_acceleration

        if remesh_velocity:
            history = (positions[1] - positions[0]) / config.dt
        else:
            history = positions[0]

        remesh_result = remesh(
            config,
            bubble_volume(positions[0], source_faces).item(),
            positions[1],
            history,
            source_faces,
            source_connect,
        )

        if remesh_velocity:
            velocities = remesh_result.history
        else:
            velocities = (
                remesh_result.positions - remesh_result.history
            ) / config.dt

        if target_acceleration:
            # a_2 = v_2 - v_1 = x_2 - 2x_1 - x_0
            target_tensor = (
                positions[2]
                - remesh_result.positions
                - (velocities * config.dt)
            )
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

        node_attr = torch.tensor(
            np.column_stack((velocities, la.norm(velocities, axis=1))),
            dtype=torch.float32,
        )

        self.add_node_set(
            "bubble", NodeSet({"velocity": NodeAttribute(node_attr)})
        )

        node_labels = torch.tensor(
            target_tensor,
            dtype=torch.float32,
        )

        self.add_label_tensor("target", node_labels)

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

    def update(self, delta: Tensor, do_remesh: bool = True) -> "Bubble":
        old_pos = self.positions.to(self.device)

        velocity = (
            self.node_sets["bubble"]["velocity"].attr[:, :3]
            + delta / self._config.dt
            if self.target_acceleration
            else delta / self._config.dt
        )
        new_pos = old_pos + velocity * self._config.dt

        if do_remesh:
            history = velocity.double() if self.remesh_velocity else old_pos

            result = remesh(
                self._config,
                self.volume().item(),
                np.ascontiguousarray(new_pos.detach().cpu().numpy()),
                np.ascontiguousarray(history.detach().cpu().numpy()),
                np.ascontiguousarray(
                    self.faces.detach()
                    .type(torch.int32)
                    .cpu()
                    .numpy()
                    .transpose()
                ),
                np.ascontiguousarray(
                    self.connect.detach().cpu().numpy().transpose()
                ),
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
                    result.history, dtype=torch.float32, device=self.device
                )
                if self.remesh_velocity
                else torch.tensor(
                    (result.positions - result.history) / self._config.dt,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        self.positions = new_pos
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

    def __init__old(
        self,
        config: SimulationConfig,
        use_remesh: bool,
        init_vol: float,
        prev: BinFile,
        data: BinFile,
        label: BinFile,
    ):
        super().__init__()

        if use_remesh:
            # Compute velocities using the remesher.
            remesh_result = remesh(
                config,
                init_vol,
                data.Bubbles[0].positions,
                # prev.Bubbles[0].positions,
                (
                    (data.Bubbles[0].positions + data.origin)
                    - (prev.Bubbles[0].positions + prev.origin)
                )
                / config.dt,
                data.Bubbles[0].faces,
                data.Bubbles[0].connect,
            )

            velocities = remesh_result.history
            # velocities = (
            #     (remesh_result.positions - remesh_result.prev_positions)
            #     + (data.origin - prev.origin)
            # ) / config.dt

            # label velocities
            label_tensor = (label.Bubbles[0].positions + label.origin) - (
                remesh_result.positions + data.origin
            )

            # Use remeshed mesh faces
            self.faces = torch.tensor(
                remesh_result.faces.transpose(), dtype=torch.long
            )

            # Use remeshed connections
            self.connect = torch.tensor(
                remesh_result.connect.transpose(), dtype=torch.long
            )

            # Store positions
            self.positions = torch.tensor(
                remesh_result.positions + data.origin, dtype=torch.float32
            )
            # self.prev_positions = torch.tensor(
            #     remesh_result.prev_positions + prev.origin, dtype=torch.float32
            # )
        else:
            velocities = (data.Bubbles[0].positions + data.origin) - (
                prev.Bubbles[0].positions + prev.origin
            ) / config.dt

            # Compute label velocities
            label_tensor = (label.Bubbles[0].positions + label.origin) - (
                data.Bubbles[0].positions + data.origin
            )

            # Use original faces
            self.faces = torch.tensor(
                data.Bubbles[0].faces.transpose(), dtype=torch.long
            )

            # Use original connections
            self.connect = torch.tensor(
                data.Bubbles[0].connect.transpose(), dtype=torch.long
            )

            # Store positions
            self.positions = torch.tensor(
                data.Bubbles[0].positions + data.origin, dtype=torch.float32
            )

        node_attr = torch.tensor(
            np.column_stack((velocities, la.norm(velocities, axis=1))),
            dtype=torch.float32,
        )

        self.add_node_set(
            "bubble", NodeSet({"velocity": NodeAttribute(node_attr)})
        )

        # Compute accelerations to serve as labels
        # self.y = torch.tensor(
        #     (label.Bubbles[0].positions + label.origin)
        #     - 2 * (remesh_result.positions + data.origin)
        #     + (remesh_result.prev_positions + prev.origin),
        #     dtype=torch.float32,
        # )
        # Compute velocities to serve as labels
        node_labels = torch.tensor(
            label_tensor,
            dtype=torch.float32,
        )

        self.add_label_tensor("velocity", node_labels)

        # Mesh -> Graph
        mesh_index = SortedIndex(
            face_to_edge(self.faces, self.positions.size()[0])
        )

        # Compute mesh edge attributes.
        source, target = mesh_index.select_single(
            self.positions, self.positions
        )
        diffs = target - source

        mesh_edge_attr = torch.cat(
            (diffs, torch.norm(diffs, dim=1).unsqueeze(1)), 1
        )

        self.add_edge_set(
            "mesh",
            EdgeSet(mesh_index, mesh_edge_attr, "bubble", "bubble"),
        )

        # Construct fluid edge set.
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
