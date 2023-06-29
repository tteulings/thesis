from typing import List

from stl import mesh
import numpy as np

import torch
from torch import Tensor

from tggnn.data.abstract.index import Index
from tggnn.data.typed_graph import EdgeSet, NodeAttribute, NodeSet, TypedGraph

from .util import NodeType


class Flag(TypedGraph):
    def __init__(
        self,
        world_pos: List[Tensor],
        faces: Tensor,
        mesh_index: Index,
        mesh_pos: Tensor,
        node_type: Tensor,
    ) -> None:
        super().__init__()

        self.positions = world_pos[1]
        self.node_type = node_type

        self.faces = faces

        vel = world_pos[1] - world_pos[0]
        self.add_node_set(
            "flag",
            NodeSet(
                {
                    "velocity": NodeAttribute(
                        torch.cat(
                            (vel, torch.norm(vel, dim=1, keepdim=True)), dim=1
                        )
                    )
                }
            ),
        )

        target_acc = world_pos[2] - 2 * world_pos[1] + world_pos[0]
        self.add_label_tensor("acceleration", target_acc)

        # world_pos_pairs = world_pos[1][mesh_index.transpose(0, 1)]
        # world_pos_diffs = world_pos_pairs[:, 0] - world_pos_pairs[:, 1]

        world_source, world_target = mesh_index.select_single(
            world_pos[1], world_pos[1]
        )
        world_pos_diffs = world_target - world_source

        # mesh_pos_pairs = mesh_pos[mesh_index.transpose(0, 1)]
        # mesh_pos_diffs = mesh_pos_pairs[:, 0] - mesh_pos_pairs[:, 1]

        # TODO: This is not very useful as the mesh is static.
        mesh_source, mesh_target = mesh_index.select_single(mesh_pos, mesh_pos)
        mesh_pos_diffs = mesh_target - mesh_source

        mesh_attr = torch.cat(
            (
                world_pos_diffs,
                torch.norm(world_pos_diffs, dim=1, keepdim=True),
                mesh_pos_diffs,
                torch.norm(mesh_pos_diffs, dim=1, keepdim=True),
            ),
            dim=1,
        )
        self.add_edge_set(
            "mesh", EdgeSet(mesh_index, mesh_attr, "flag", "flag")
        )

    def update(self, delta: Tensor) -> "Flag":
        self.positions = self.positions.to(self.device)

        self.node_type = self.node_type.to(self.device)
        mask = self.node_type == NodeType.NORMAL

        delta = delta.where(mask, torch.zeros_like(delta))

        velocity = self.node_sets["flag"]["velocity"]

        # Update the velocity
        velocity.attr[:, :3] += delta
        velocity.attr[:, 3] = torch.norm(velocity.attr[:, :3], dim=1)

        # Update the position
        self.positions += velocity.attr[:, :3]

        mesh = self.edge_sets["mesh"]

        world_source, world_target = mesh.index.select_single(
            self.positions, self.positions
        )
        world_pos_diffs = world_target - world_source

        if mesh.attr is not None:
            mesh.attr[:, :4] = torch.cat(
                (
                    world_pos_diffs,
                    torch.norm(world_pos_diffs, dim=1, keepdim=True),
                ),
                dim=1,
            )

        return self

    def as_stl(self) -> mesh.Mesh:
        # Set up the mesh
        flagMesh = mesh.Mesh(
            np.zeros(self.faces.shape[1], dtype=mesh.Mesh.dtype)
        )

        # Set up the mesh position vectors
        flagMesh.vectors = (
            self.positions[self.faces.transpose(0, 1)].detach().cpu().numpy()
        )

        return flagMesh
