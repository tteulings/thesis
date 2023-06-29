from typing import Tuple

import torch
import torch.linalg as la
from torch import Tensor

from scipy.special import sph_harm

from ..remesh.bubble import Bubble


def get_mlk(l_max: int) -> Tuple[Tensor, Tensor, int]:
    k = (l_max + 1) ** 2

    j = torch.arange(1, k + 1)

    degrees = torch.arange(0, l_max + 1)
    l = degrees.repeat_interleave(degrees * 2 + 1)

    m = j - (l ** 2 + l + 1)

    return m, l, k


def get_harmonics(azimuthal: Tensor, polar: Tensor, l_max: int) -> Tensor:
    m, l, k = get_mlk(l_max)

    harmonics: Tensor = sph_harm(
        m,
        l,
        azimuthal.unsqueeze(1).expand(-1, k),
        polar.unsqueeze(1).expand(-1, k),
    ).real

    return harmonics


# NOTE: Based on https://doi.org/10.1155/2015/582870.
def harmonic_decomposition(
    bubble: Bubble, l_max: int, nu: float
) -> Tuple[Tensor, Tensor]:
    center = bubble.centroid()
    centered = bubble.positions - center

    # Compute spherical coordinates
    r = la.vector_norm(centered, dim=1)
    azim = torch.atan2(centered[:, 1], centered[:, 0])
    polar = torch.acos(centered[:, 2] / r)

    k = (l_max + 1) ** 2

    j = torch.arange(1, k + 1)

    degrees = torch.arange(0, l_max + 1)
    l = degrees.repeat_interleave(degrees * 2 + 1)

    m = j - (l ** 2 + l + 1)

    # Compute spherical harmonics for all (m, l) pairs up to l_max.
    harmonics: Tensor = sph_harm(
        m,
        l,
        azim.unsqueeze(1).expand(-1, k),
        polar.unsqueeze(1).expand(-1, k),
    ).real

    # Apply Tikhonov regularization using the Laplace-Beltrami operator.
    tikhonov = torch.diag(l ** 2 * (l ** 2 + 1) ** 2)

    A = torch.matmul(harmonics.transpose(0, 1), harmonics) + nu * tikhonov
    b = torch.matmul(harmonics.transpose(0, 1), r)

    # Compute harmonic weights using a least squares fit.
    params = la.lstsq(A, b)

    return harmonics, params.solution
