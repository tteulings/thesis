import os
from copy import deepcopy

import torch
from torch.nn.functional import mse_loss

from ..common import Checkpoint
from .args import flag_model_args
from .data.dataset import FlagDataset
from .impl.model import flag_model

args = flag_model_args(False)
sequence_len = 399

dataset = FlagDataset(args.data_directory)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = flag_model(
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

    loss = torch.zeros(sequence_len, dtype=torch.float64, device=device)

    sequence_count = 0
    print_seq = 0

    for seq_num in range(0, args.num_rollouts):
        init_id = seq_num * sequence_len

        flag = dataset[init_id]
        flag.to(device)

        sequence_count += 1

        for id in range(init_id, init_id + sequence_len - 1):
            out = model(deepcopy(flag))

            delta = model._label_normalizers["acceleration"].inverse(
                out.node_sets["flag"]["velocity"].attr
            )

            flag.update(delta)

            next = dataset[id + 1]
            next.to(device)
            next.positions = next.positions.to(device)

            loss[id - init_id] += mse_loss(flag.positions, next.positions)

            if seq_num == print_seq:
                flag.as_stl().save(
                    os.path.join(args.output_directory, f"pred_{id}.stl")
                )
                next.as_stl().save(
                    os.path.join(args.output_directory, f"truth_{id}.stl")
                )

    print((loss / args.num_rollouts).cpu().numpy())
