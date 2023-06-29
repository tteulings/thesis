import os
from copy import deepcopy
from math import floor

import torch
from torch.nn.functional import mse_loss

from tggnn.network.loss import point_to_surface_loss

from models.bubble.args import bubble_model_args
from models.bubble.impl.basic import bubble_basic_model
from models.common import Checkpoint

from .dataset import BubbleDataset

args = bubble_model_args(False)
sequence_len = 8000

dataset = BubbleDataset(
    args.data_directory,
    remesh_velocity=args.remesh_velocity,
    target_acceleration=args.target_acceleration,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = bubble_basic_model(
    dataset.layout(), args.latent_size, args.num_layers, args.iterations
).to(device)
model.eval()

if os.path.exists(args.checkpoint):
    checkpoint: Checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.model_state_dict)
else:
    raise Exception(f'Could not load checkpoint: "{args.checkpoint}".')

with torch.no_grad():
    first = True

    trans_loss = torch.zeros(args.steps, dtype=torch.float64, device=device)
    deform_loss = torch.zeros(args.steps, dtype=torch.float64, device=device)

    sequence_count = 0
    print_seq = 5

    for init_id in range(100, sequence_len, args.steps):
        if init_id + args.steps >= sequence_len:
            break

        print(init_id)
        bubble = dataset[init_id]
        bubble.to(device)

        sequence_count += 1

        for id in range(init_id, init_id + args.steps):
            out = model(deepcopy(bubble))

            delta = model._label_normalizers["target"].inverse(
                out.node_sets["bubble"]["velocity"].attr
            )

            bubble.update(delta, True)

            # if first:
            #     print(f"One step MSE: {mse_loss(delta, bubble.labels['target'])}")
            #     print(
            #         "Relative error:",
            #         (
            #             (delta - bubble.labels["target"]).abs()
            #             / bubble.labels["target"].abs()
            #         ).mean(dim=0),
            #     )

            #     first = False

            next = dataset[id + 1]
            next.to(device)
            next.positions = next.positions.to(device)
            next.faces = next.faces.to(device)

            deform_loss[id - init_id] += point_to_surface_loss(
                (bubble.positions - bubble.centroid()).float(),
                (next.positions - next.centroid()).float(),
                next.faces,
            )
            trans_loss[id - init_id] += mse_loss(
                bubble.centroid(), next.centroid()
            )

            if sequence_count == print_seq:
                bubble.as_stl().save(
                    os.path.join(args.output_directory, f"pred_{id}.stl")
                )
                next.as_stl().save(
                    os.path.join(args.output_directory, f"truth_{id}.stl")
                )

    print(
        (
            torch.cat(
                (trans_loss.unsqueeze(1), deform_loss.unsqueeze(1)), dim=1
            )
            / floor((sequence_len - 100) / args.steps)
        )
        .cpu()
        .numpy()
    )
