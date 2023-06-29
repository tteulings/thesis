import os
import sys
from timeit import default_timer

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from tggnn.data.typed_graph import TypedGraph
from tggnn.model.stateful import StatefulModel

from ...common import Checkpoint
from ..args import bubble_model_args
from ..data import Bubble, BubbleSequenceDataset
from ..data.transforms import AdditiveNoiseTransform
from ..impl.memory import bubble_memory_model, center_memory


# Prepare the argument parser
args = bubble_model_args(True)

# Load the bubble trajectory dataset.
dataset = BubbleSequenceDataset(
    args.data_directory,
    transforms=[AdditiveNoiseTransform(args.noise)]
    if args.noise is not None
    else None,
    remesh_velocity=args.remesh_velocity,
    target_acceleration=args.target_acceleration,
)

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
    dataset, batch_size=None, shuffle=True, num_workers=0
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

# Run model epochs
def train(
    model: StatefulModel[Bubble, TypedGraph],
    state: TypedGraph,
    gather_statistics: bool,
) -> None:
    count = 0
    train_loss = 0

    epoch_start = model.epoch.item()

    while True:
        for sequence in sequence_loader:
            bubble_loader = DataLoader(
                sequence, batch_size=None, shuffle=False, num_workers=4
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
            # losses = []

            for bubble in bubble_loader:
                bubble.to(device)

                state = center_memory(bubble, state)

                out, state = model.forward(bubble, state)

                pred = out.node_sets["bubble"]["velocity"].attr
                label = out.labels["target"]

                loss += F.mse_loss(pred, label)
                # losses.append(loss)

                count += 1
                model.epoch += 1

                # TBPTT with k1 == k2
                if count % args.tbptt == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    # for i, loss in reversed(list(enumerate(losses))):
                    #     loss.backward(retain_graph=i > 0)

                    optimizer.step()
                    # scheduler.step()
                    train_loss += float(loss)

                    # TODO: I would rather use the inplace `detach_()`. However, I
                    # have a feeling that it doesn't have the same effect on the
                    # autograd graph. Find out if there are any differences.
                    centroid["memory"].attr = centroid["memory"].attr.detach()
                    # losses = []
                    loss.detach_().zero_()

                if count % args.print_every == 0:
                    print(
                        f"{model.epoch.item():<7}{train_loss/count:<18.8e}"
                        f"{default_timer() - timer}"
                    )
                    sys.stdout.flush()

                    train_loss = 0
                    count = 0

                    if args.checkpoint is not None:
                        torch.save(
                            Checkpoint(
                                model.state_dict(), optimizer.state_dict(), None
                            ),
                            args.checkpoint,
                        )

                    timer = default_timer()

                if model.epoch - epoch_start >= args.epochs:
                    return


train(model, state, gather_statistics)
