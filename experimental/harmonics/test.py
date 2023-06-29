from ..remesh.dataset import BubbleDataset
from .decompose import harmonic_decomposition

import torch.linalg as la

import torch


dataset = BubbleDataset("/home/daan/git/bubble-data/n50")
bubble = dataset[6000]

l_max = 20

with torch.no_grad():
    harmonics, params = harmonic_decomposition(bubble, l_max, 1e-3)
    
    r_pred = torch.matmul(harmonics, params)
    
    center = bubble.centroid()
    centered = bubble.positions - center
    
    # Compute spherical coordinates
    r = la.vector_norm(centered, dim=1)
    azimuthal = torch.atan2(centered[:, 1], centered[:, 0])
    polar = torch.acos(centered[:, 2] / r)
    
    x = r_pred * azimuthal.cos() * polar.sin()
    y = r_pred * azimuthal.sin() * polar.sin()
    z = r_pred * polar.cos()
    
    bubble.positions = centered
    bubble.as_stl().save("before.stl")
    bubble.positions = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
    bubble.as_stl().save("after.stl")
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(x, y, z)
    # ax.scatter(centered[:, 0], centered[:, 1], centered[:, 2])
    # plt.show()
    
    # r = la.vector_norm(bubble.positions - bubble.centroid(), dim=1)
    # r_pred = torch.matmul(harmonics, params)
    
    # rel_error = torch.mean((r - r_pred).abs() / r.abs())
    # print(rel_error.item())
