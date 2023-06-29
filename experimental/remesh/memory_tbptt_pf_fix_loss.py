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


# Prepare the argument parser
args = bubble_model_args(True)
print(args)
# Load the bubble trajectory dataset.
dataset = BubbleSequenceDataset(
    args.data_directory,
    # transforms=[RelativeNoiseTransform(args.noise)]
    # if args.noise is not None
    # else None,
    remesh_velocity=args.remesh_velocity,
    target_acceleration=args.target_acceleration,
    # ignore_sequences=[1, 2,3,4,5,6,7,8]
    # start_point=1
)
transform = RelativeNoiseTransform(args.noise) if args.noise is not None else None
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

gather_statistics = True

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

# Gather normalizer statistics
model.train()
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
        for sequence in sequence_loader:
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
     

                if count == 0:
                    if transform is not None:
                        current = transform(bubble)
                    else:
                        current = bubble
                elif (count) % args.tbptt == 0:
                    # print('new',count)

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

                # print(pred,)
                # print(model._label_normalizers["target"].inverse(label),)

                current.update(
                    model._label_normalizers["target"].inverse(label), False
                )

                # one_step_loss = F.mse_loss(pred, label)
                # loss += one_step_loss
                # train_loss += float(one_step_loss)



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
                # print(count,args.tbptt, count%args.tbptt)
                # TBPTT with k1 == k2
                if (count-1) % (args.tbptt) == 0 and count > 1:
                    # print('backprop',count)
                    one_step_loss = F.mse_loss(pred, label)
                    loss += one_step_loss
                    train_loss += float(one_step_loss)

                    optimizer.zero_grad()
                    one_step_loss.backward()

                    optimizer.step()
                    # scheduler.step()
                    # TODO: I would rather use the inplace `detach_()`. However, I
                    # have a feeling that it doesn't have the same effect on the
                    # autograd graph. Find out if there are any differences.

                    print(loss)
                    loss.detach_().zero_()
                    one_step_loss.detach_().zero_()
                else:
                    loss += F.mse_loss(pred, label).detach()
                centroid["memory"].attr = centroid["memory"].attr.detach()
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
