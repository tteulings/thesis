from copy import deepcopy
import os
import sys
from timeit import default_timer

import numpy as np

import os;

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from tggnn.data.typed_graph import TypedGraph
from tggnn.model.stateful import StatefulModel

from models.common import Checkpoint
from models.bubble.args import bubble_model_args
from models.bubble.impl.memory import bubble_memory_model, center_memory

from .bubble import Bubble
from .sequence_dataset import BubbleSequenceDataset
from .transforms import AdditiveNoiseTransform, RelativeNoiseTransform

import wandb
# Prepare the argument parser
args = bubble_model_args(True)
print(args)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="thesis-tteulings",
    
    # track hyperparameters and run metadata
    config={
    "architecture": "Fixed predicted input, only backpropagate end",
    **vars(args),
    }
)
# Load the bubble trajectory dataset.
dataset = BubbleSequenceDataset(
    args.data_directory,
    # transforms=[RelativeNoiseTransform(args.noise)]
    # if args.noise is not None
    # else None,
    remesh_velocity=args.remesh_velocity,
    target_acceleration=args.target_acceleration,
    # ignore_sequences=[1, 2,3,4,5,6,7,8]
    start_point=2
)
transform = AdditiveNoiseTransform(args.noise) if args.noise is not None else None
# A small epsilon value to stabilize ARE computation.
eps = 1e-8

# Set the torch seed
# torch.manual_seed(23)

# Setup the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, state = bubble_memory_model(
    dataset.layout(), args.latent_size, args.num_layers, args.iterations
)

model.to(device)
state.to(device)

optimizer = Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=5e-4, amsgrad=False
)
scheduler = ExponentialLR(optimizer, gamma=(1e-2) ** (1 / 5e6))

# Split the dataset, into norm, hidden, and training segments.
sequence_loader = DataLoader(
    dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=True
)

gather_statistics = False

if args.checkpoint is not None:
    if os.path.exists(args.checkpoint):
        checkpoint: Checkpoint = torch.load(args.checkpoint)

        model.load_state_dict(checkpoint.model_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        if checkpoint.scheduler_state_dict is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state_dict)

        gather_statistics = False

# Print header
print("{:-<80}".format(""))
print("Epoch{:2}Train loss{:8}Time".format(" ", " "))
print("{:-<80}".format(""))
torch.cuda.empty_cache()
# Gather normalizer statistics
model.train()

wandb.watch(model, log_freq=10, log='all')

# print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
# Run model epochs
def train(
    model: StatefulModel[Bubble, TypedGraph],
    state: TypedGraph,
    gather_statistics: bool,
) -> None:
    train_loss = 0
    rel_error = np.array([0, 0, 0], dtype=np.float32)
    epoch_start = model.epoch.item()

    while True:
        # print()
        for si, sequence in enumerate(sequence_loader):
            # print()
            # print(sequence)
            bubble_loader = DataLoader(
                sequence, batch_size=None, shuffle=False, num_workers=8, pin_memory=True
            )

            if gather_statistics:
                for i, bubble in enumerate(bubble_loader):
                    bubble.to(device)
                    state = center_memory(bubble, state)

                    model.gather_statistics(bubble, state)

                    if i >= 1000:
                        break

                gather_statistics = False

            # Initialize/reset hidden state
            centroid = state.node_sets["centroid"]
            centroid["memory"].attr.zero_()

            timer = default_timer()

            loss = torch.tensor([0.0], device=device)

            current = None



            # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
            for (count, bubble) in enumerate(bubble_loader):
                # print(count)

                if count == 0:
                    u = np.random.choice(list(range(int(args.tbptt*0.1), args.tbptt))   )
                    if transform is not None:
                        current = transform(bubble)
                    else:
                        current = bubble
                elif count >  0 and (count - 1) % u == 0:
                    print(count,'new_bubble')
                    u = np.random.choice(list(range(int(args.tbptt*0.1), args.tbptt))   )

                    if transform is not None:
                        current = transform(bubble)
                    else:
                        current = bubble
                else:
                    if current is None:
                        # Just to satisfy the type system.
                        # This should never happen.
                        return

                    current.remesh(bubble.instructions)
                    current.labels["target"] = bubble.labels["target"]
                # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                current.to(device)

                state = center_memory(current, state)

                out, state = model.forward(current, state)
                # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                pred = out.node_sets["bubble"]["velocity"].attr
                label = out.labels["target"]


                error = ((label-pred)**2).mean(axis=0)
                # print(error)
                current_centroid = current.centroid()
                # current.update(
                #     model._label_normalizers["target"].inverse(label), False
                # )

                one_step_loss = F.mse_loss(pred, label)
                loss += one_step_loss
                train_loss += float(one_step_loss)



                rel_error += (
                    torch.mean(
                        torch.abs(pred - label)
                        / torch.abs(label).clamp_min(eps),
                        0,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                model.epoch += 1
                # print(torch.mean(pred, dim=0))


                current.update(
                    model._label_normalizers["target"].inverse(pred.detach()), False
                )
                vel_pred = current.centroid() - current_centroid

                bubble_centroid = bubble.centroid()
                # # print(label)
                # # current_update = deepcopy(current)
                bubble.update(
                    model._label_normalizers["target"].inverse(label).to(bubble.device), False
                )


                # preds = model._label_normalizers["target"].inverse(pred)
                vel_true = bubble.centroid() - bubble_centroid

                # wandb.log({'velocity_centroid_x': vel_pred[ 0], 'velocity_centroid_y': vel_pred[1], 'velocity_centroid_z': vel_pred[ 2], 'true_velocity_centroid_x': vel_true[ 0], 'true_velocity_centroid_y': vel_true[1], 'true_velocity_centroid_z': vel_true[ 2]})
                # wandb.log({'epoch': model.epoch, 'bubble_num': si+1, "loss": loss})
                wandb.log({'u': u, 'loss_x': error[0], 'loss_y': error[1], 'loss_z': error[2], 'epoch': model.epoch, 'bubble_num': si+1, "loss": loss, 'onesteploss': one_step_loss , 'velocity_centroid_x': vel_pred[ 0], 'velocity_centroid_y': vel_pred[1], 'velocity_centroid_z': vel_pred[ 2], 'true_velocity_centroid_x': vel_true[ 0], 'true_velocity_centroid_y': vel_true[1], 'true_velocity_centroid_z': vel_true[ 2]})

                # print(count,args.tbptt, count%args.tbptt)
                # TBPTT with k1 == k2
                if count % u == 0  :
                    
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    # scheduler.step()
                    print(count, 'backprop\n')

                if (u - (count % u)) >= 3:
                    print('detach')
                    # TODO: I would rather use the inplace `detach_()`. However, I
                    # have a feeling that it doesn't have the same effect on the
                    # autograd graph. Find out if there are any differences.
                    centroid["memory"].attr = centroid["memory"].attr.detach()
                    loss.detach_().zero_()
                    one_step_loss.detach_().zero_()
                else:
                    print(count, 'detach')

                if (count + 1) % args.print_every == 0:
              
                    print(
                        f"{model.epoch.item():<7}{train_loss/args.print_every:<18.8e}"
                        f"{f'{rel_error/args.print_every}': <37}{default_timer() - timer}"
                    )
                    sys.stdout.flush()

                    train_loss = 0
                    rel_error.fill(0)

                    if args.checkpoint is not None:
                        torch.save(
                            Checkpoint(
                                model.state_dict(), optimizer.state_dict(), None
                            ),
                            args.checkpoint,
                        )
                        if (count + 1) % 10000 == 0:
                            torch.save(
                                Checkpoint(
                                    model.state_dict(), optimizer.state_dict(), None
                                ),
                                args.checkpoint+f"_{count}",
                            )
                    timer = default_timer()

                if model.epoch - epoch_start >= args.epochs:
                    return


train(model, state, gather_statistics)
