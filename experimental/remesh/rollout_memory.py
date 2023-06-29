import os
from copy import deepcopy

import torch
from tggnn.network.loss import point_to_surface_loss
from torch.nn.functional import mse_loss

from models.common import Checkpoint
from models.bubble.args import bubble_model_args
from models.bubble.impl.memory import bubble_memory_model, center_memory

from .sequence_dataset import BubbleSequence

args = bubble_model_args(False)

sequence = BubbleSequence(
    args.data_directory,
    1,
    remesh_velocity=args.remesh_velocity,
    target_acceleration=args.target_acceleration,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, state = bubble_memory_model(
    sequence.layout(), args.latent_size, args.num_layers, args.iterations
)

model.eval()
model.to(device)
state.to(device)

if os.path.exists(args.checkpoint):
    checkpoint: Checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.model_state_dict)
else:
    raise Exception(f'Could not load checkpoint: "{args.checkpoint}".')

with torch.no_grad():
    trans_loss = torch.zeros(args.steps, dtype=torch.float64, device=device)
    deform_loss = torch.zeros(args.steps, dtype=torch.float64, device=device)

    sequence_count = 0
    print_seq = 5
    windup = 100

    for id in range(windup):
        bubble = sequence[id]
        bubble.to(device)

        _, state = model.forward(bubble, center_memory(bubble, state))

    for init_id in range(windup, len(sequence), args.steps):
        if init_id + args.steps >= len(sequence):
            break

        bubble = sequence[init_id]
        bubble.to(device)

        sequence_count += 1

        for id in range(init_id, init_id + args.steps):
            out, state = model.forward(
                deepcopy(bubble), center_memory(bubble, state)
            )

            delta = model._label_normalizers["target"].inverse(
                out.node_sets["bubble"]["velocity"].attr
            )

            bubble.update(delta, True)

            next = sequence[id + 1]
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
            / sequence_count
        )
        .cpu()
        .numpy()
    )
