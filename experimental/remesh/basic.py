import os
import sys
from timeit import default_timer

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from tggnn.model.basic import BasicModel

from models.common import Checkpoint
from models.bubble.args import bubble_model_args
from models.bubble.impl.basic import bubble_basic_model
from .bubble import Bubble
from .dataset import BubbleDataset
from .transforms import AdditiveNoiseTransform


# Prepare the argument parser
args = bubble_model_args(True)

# Load the bubble trajectory dataset.
dataset = BubbleDataset(
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

model = bubble_basic_model(
    dataset.layout(), args.latent_size, args.num_layers, args.iterations
).to(device)

optimizer = Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=5e-4, amsgrad=False
)
scheduler = ExponentialLR(optimizer, gamma=(1e-2) ** (1 / 5e6))

# Create an iterable loader for the training dataset.
bubble_loader = DataLoader(
    dataset, batch_size=None, shuffle=True, num_workers=6, pin_memory=True
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
print("Epoch{:2}Train loss{:8}Time".format(" ", " ", " "))
print("{:-<80}".format(""))

# Gather normalizer statistics
model.train()

# Run model epochs
def train(model: BasicModel[Bubble], gather_statistics: bool) -> None:
    count = 0
    train_loss = 0

    epoch_start = model.epoch.item()

    if gather_statistics:
        for i, bubble in enumerate(bubble_loader):
            bubble.to(device)

            model.gather_statistics(bubble)

            if i >= 1000:
                break

    timer = default_timer()

    while True:
        for bubble in bubble_loader:
            bubble.to(device)

            optimizer.zero_grad()

            out = model.forward(bubble)

            pred = out.node_sets["bubble"]["velocity"].attr
            label = out.labels["target"]

            loss = F.mse_loss(pred, label)

            train_loss += float(loss)
            count += 1
            model.epoch += 1

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if count % args.print_every == 0:
                print(
                    f"{model.epoch.item():<7}{train_loss/count:<18.8e}{default_timer() - timer}"
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


train(model, gather_statistics)
