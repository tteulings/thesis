import torch
# from torch.utils.data import DataLoader

from .dataset import BubbleDataset

# for bubble in DataLoader(
#     dataset, shuffle=False, batch_size=None, num_workers=0
# ):
#     print(default_timer() - t)
#     # print(bubble.volume())
#     t = default_timer()
dataset = BubbleDataset(
    "/home/daan/git/bubble-data/n10", remesh_velocity=True
)

prev = dataset[5000]
current = dataset[5001]
next = dataset[5002]

cur_vel = current.positions - prev.positions
next_vel = next.positions - current.positions

print(torch.mean(current.positions, dim=0))
print(torch.mean(cur_vel, dim=0))
print(torch.mean(next_vel - cur_vel, dim=0))
# print(bubble.node_sets["bubble"]["velocity"].attr)
# next = dataset[1]

# bubble_loader = DataLoader(
#     dataset, shuffle=False, num_workers=0, batch_size=None
# )

# for bid, bubble in enumerate(bubble_loader):
#     print(f"Loading bubble: {bid}")
#     exit()
