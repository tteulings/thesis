import os
import sys
from copy import deepcopy
import gc
import torch
from torch.nn.functional import mse_loss

from tggnn.network.loss import point_to_surface_loss

from models.bubble.args import bubble_model_args
from models.bubble.impl.memory import bubble_memory_model, center_memory
from models.common import Checkpoint

from .dataset import BubbleDataset
import wandb
args = bubble_model_args(False)



model_name = args.checkpoint.split('/')[-1]

other_runs = [i for i in range(len(os.listdir('./results'))+1) if f'{model_name}_{i}' not in os.listdir('./results')]
print(model_name,os.listdir('./results'))
postfix = 0

if other_runs != []:
    postfix = min(other_runs)



path = os.path.join(f'results/{model_name}_{postfix}')
os.mkdir(path)
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="thesis-tteulings",
    tags=['predict'],
    #  =args.data_directory.split('/')[-1],

    # track hyperparameters and run metadata
    config={
        model_name:model_name,
    "architecture": "Fixed predicted input",
    **vars(args),
    }
)
dataset = BubbleDataset(
    args.data_directory,
    remesh_velocity=args.remesh_velocity,
    target_acceleration=args.target_acceleration,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, state = bubble_memory_model(
    dataset.layout(), args.latent_size, args.num_layers, args.iterations
)

model.eval().to(device)
state.to(device)

if os.path.exists(args.checkpoint):
    checkpoint: Checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.model_state_dict)
else:
    raise Exception(f'Could not load checkpoint: "{args.checkpoint}".')

with torch.no_grad():
    for id in range(args.bubble_id - 0, args.bubble_id):
        print(f"Preparing hidden: {id}")
        bubble = dataset[id]
        bubble.to(device)

        _, state = model.forward(bubble, center_memory(bubble, state))

    bubble = dataset[args.bubble_id]
    bubble.to(device)

    first = True

    for id in range(args.bubble_id, args.bubble_id + args.steps):
        gc.collect()
        c_bubble_og = bubble.centroid()

        
        out, state = model.forward(
            deepcopy(bubble), center_memory(bubble, state)
        )

        delta = model._label_normalizers["target"].inverse(
            out.node_sets["bubble"]["velocity"].attr
        )

        bubble.update(delta, True)

        if first:
            print(f"One step MSE: {mse_loss(delta, bubble.labels['target'])}")
            print(
                "Relative error:",
                (
                    (delta - bubble.labels["target"]).abs()
                    / bubble.labels["target"].abs()
                ).mean(dim=0),
            )

            first = False

        next = dataset[id + 1]
        next.to(device)

        next.positions = next.positions.to(device)
        next.faces = next.faces.to(device)

        c_bubble = bubble.centroid()
        c_next = next.centroid()

        # c_vel = (c_next - c_bubble) / dataset._config.dt
        # c_vel = bubble.node_sets["bubble"]["velocity"].attr.mean(dim=0)
        c_vel = c_bubble - c_bubble_og.to(c_bubble.device)
        c_true_vel = dataset[id + 1].centroid() - dataset[id].centroid()

        pts_loss = point_to_surface_loss(
            (bubble.positions - c_bubble).float(),
            (next.positions - c_next).float(),
            next.faces,
        )

        trans_loss = mse_loss(c_bubble, c_next)

        wandb.log({'id': id, 'p2s_loss':pts_loss, 'trans_loss': trans_loss, 'c_true_vel_x': c_true_vel[0], 'c_true_vel_y': c_true_vel[1], 'c_true_vel_z': c_true_vel[2], 'c_veloc_x': c_vel[0], 'c_veloc_y': c_vel[1], 'c_veloc_z': c_vel[2]})
        print(
            f"{id * dataset._config.dt},{trans_loss.item()},{pts_loss.item()},"
            f"{c_bubble[0].item()},{c_bubble[1].item()},{c_bubble[2].item()},"
            f"{c_vel[0].item()},{c_vel[1].item()},{c_vel[2].item()}"
        )
        sys.stdout.flush()
        bubble.as_stl().save(
            path + f"/pred_{id}.stl"
        )
        next.as_stl().save(
            path+ f"/truth_{id}.stl"
        )
