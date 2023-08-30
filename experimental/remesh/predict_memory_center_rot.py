import os
import sys
import numpy as np
from copy import deepcopy
import gc
import torch
from torch.nn.functional import mse_loss

from tggnn.network.loss import point_to_surface_loss

from models.bubble.args import bubble_model_args
from models.bubble.impl.memory import bubble_memory_model, center_memory
from models.common import Checkpoint


from .bubble import Bubble, rotation_matrix_to_y_axis, rotation_matrix_to_diag

from .dataset import BubbleDataset
import wandb

from loky import get_reusable_executor


executor = get_reusable_executor(max_workers=1, timeout=2)

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
    center_prediction = True
)
def load_data(bubble_id, steps):
    all_bubbles = []
    dataset = BubbleDataset(
        args.data_directory,
        remesh_velocity=args.remesh_velocity,
        target_acceleration=args.target_acceleration,
        center_prediction = True
    )
    for id in range(args.bubble_id, args.bubble_id + args.steps):
        all_bubbles.append(dataset[args.bubble_id])

    return all_bubbles

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = 'cpu'
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


center_normalizer =  torch.tensor([.7*10**7, .35*10**7, .5*10**5], device='cuda:0')

center_normalizer =  torch.tensor([1*10**6, .1*10**7, 1*10**6])
center_normalizer =  torch.tensor([1*10**6, .1*10**7, .8*10**6])


dataset.set_rotation_matrix(np.diag([1.0,1.0,1.0]))
# rotation_matrix = rotation_matrix_to_diag(sequence[0].center_velocity)
print(dataset[0].center_velocity)
rotation_matrix = rotation_matrix_to_y_axis(dataset[0].center_velocity)
# rotation_matrix = rotation_matrix_to_diag(dataset[args.bubble_id - 100].center_velocity)
print(rotation_matrix)
dataset.set_rotation_matrix(rotation_matrix)
centroid = state.node_sets["centroid"]


centroid["memory"].attr = torch.ones_like(centroid["memory"].attr)


with torch.no_grad():
    for id in range(args.bubble_id - 0, args.bubble_id):
        print(f"Preparing hidden: {id}")
        bubble = dataset[id]
        bubble.to(device)

#         print(bubble.center_velocity.to(device).reshape((1,3))*center_normalizer.to(device), bubble.center_target.to(device).reshape((1,3))
# )
#         print(bubble.center_velocity)
        if id  == 0:
            state.node_sets['centroid']['velocity'].attr = bubble.center_velocity.to(device).reshape((1,3))
        # print(state.node_sets['centroid']['velocity'].attr)
        out, state, center_out = model.forward(bubble, center_memory(bubble, state))

        print(center_out, bubble.center_target.to(device).reshape((1,3))*center_normalizer.to(device), bubble.center_velocity)
        state.node_sets['centroid']['velocity'].attr = bubble.center_target.to(device).reshape((1,3))*center_normalizer.to(device).reshape(1,3)
        print()
        # break
    

    bubble = dataset[args.bubble_id]
    bubble.to(device)

    first = True
    state.node_sets['centroid']['velocity'].attr = bubble.center_velocity.to(device).reshape((1,3))*center_normalizer.to(device).reshape(1,3)

    for id in range(args.bubble_id, args.bubble_id + args.steps):
        gc.collect()
        c_bubble_og = bubble.centroid()

        print(id)
        # print(centroid['velocity'].attr, centroid["memory"].attr)

    
        out, state, center_out = model.forward(
            deepcopy(bubble), center_memory(bubble, state)
        )

        print(center_out, bubble.center_target.to(device).reshape((1,3))*center_normalizer.to(device), bubble.center_velocity)

        delta = model._label_normalizers["target"].inverse(
            out.node_sets["bubble"]["velocity"].attr
        )

        bubble.update(delta, True, center_out/center_normalizer.to(device))
        state.node_sets['centroid']['velocity'].attr = bubble.center_velocity.to(device).reshape((1,3))*center_normalizer.to(device).reshape(1,3)
        print(center_out, bubble.center_target.to(device).reshape((1,3))*center_normalizer.to(device), bubble.center_velocity)
        # if id > 1:
        #     break

        if first:
            # print(f"One step MSE: {mse_loss(delta, bubble.labels['target'])}")
            # print(
            #     "Relative error:",
            #     (
            #         (delta - bubble.labels["target"]).abs()
            #         / bubble.labels["target"].abs()
            #     ).mean(dim=0),
            # )

            first = False

        next = dataset[id + 1]
        next.to(device)

        next.positions = next.positions.to(device)
        next.faces = next.faces.to(device)

        c_bubble = bubble.centroid()
        c_next = next.centroid()
        print('center', bubble.centroid(), c_next)

        # c_vel = (c_next - c_bubble) / dataset._config.dt
        # c_vel = bubble.node_sets["bubble"]["velocity"].attr.mean(dim=0)
        c_vel = c_bubble - c_bubble_og.to(c_bubble.device)
        c_true_vel = dataset[id + 1].centroid() - dataset[id].centroid()
        c_true_vel =  dataset[id].center_target
        pts_loss = point_to_surface_loss(
            (bubble.positions - c_bubble).float(),
            (next.positions - c_next).float(),
            next.faces,
        )

        trans_loss = mse_loss(c_bubble, c_next)
        center_out/=center_normalizer.to(device)
        wandb.log({'id': id, 'p2s_loss':pts_loss, 'trans_loss': trans_loss, 'c_true_vel_x': c_true_vel[0], 'c_true_vel_y': c_true_vel[1], 'c_true_vel_z': c_true_vel[2], 'c_veloc_x': c_vel[0], 'c_veloc_y': c_vel[1], 'c_veloc_z': c_vel[2], 'c_velo_x': center_out[0][0], 'c_velo_y': center_out[0][1], 'c_velo_z': center_out[0][2]})
        print(
            f"{id * dataset._config.dt},{trans_loss.item()},{pts_loss.item()},"
            f"{c_bubble[0].item()},{c_bubble[1].item()},{c_bubble[2].item()},"
            f"{c_vel[0].item()},{c_vel[1].item()},{c_vel[2].item()}"
        )
        print()
        sys.stdout.flush()
        bubble.as_stl().save(
            path + f"/pred_{id}.stl"
        )
        next.as_stl().save(
            path+ f"/truth_{id}.stl"
        )
