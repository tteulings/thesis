from typing import Optional

import os
import os.path as osp
import errno

import torch
from torch import Tensor


def unique_index(edge_index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    num_nodes = (
        num_nodes if num_nodes is not None else int(edge_index.max() + 1)
    )

    # NOTE: Always use int64 as the pair indicators can become quite large.
    idx = torch.empty(
        edge_index.size(1) + 1, device=edge_index.device, dtype=torch.int64
    )

    idx[0] = -1
    idx[1:] = edge_index[1]

    # NOTE: This will generate a unique value for each node index pair.
    idx[1:].mul_(num_nodes).add_(edge_index[0])

    idx[1:], perm = idx[1:].sort()
    edge_index = edge_index[:, perm]

    mask = idx[1:] > idx[:-1]

    if mask.all():
        return edge_index

    edge_index = edge_index[:, mask]

    return edge_index


def to_undirected(
    edge_index: Tensor, num_nodes: Optional[int] = None
) -> Tensor:
    return unique_index(
        torch.cat((edge_index, edge_index.flip(dims=[0])), dim=1), num_nodes
    )


def face_to_edge(faces: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    edge_index = torch.cat([faces[:2], faces[1:], faces[::2]], dim=1)
    return to_undirected(edge_index, num_nodes)


def makedirs(path: str) -> None:
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as err:
        if err.errno != errno.EEXIST and osp.isdir(path):
            raise err
