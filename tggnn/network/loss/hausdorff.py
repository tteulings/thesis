import torch
from torch import Tensor

from pytorch3d.loss.point_mesh_distance import point_face_distance


# TODO: Implement symmetrical hausdorff distance.
def hausdorff_loss(
    source_pos: Tensor, target_pos: Tensor, target_faces: Tensor
) -> Tensor:
    if not (source_pos.device == target_pos.device == target_faces.device):
        raise Exception(f"All tensors should be on the same device.")

    device = source_pos.device

    return point_face_distance(
        source_pos,
        torch.zeros(1, dtype=torch.long, device=device),
        target_pos[target_faces.transpose(0, 1)],
        torch.zeros(1, dtype=torch.long, device=device),
        source_pos.size()[0],
    ).max()
