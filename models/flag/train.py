from typing import cast

import os
import sys
from timeit import default_timer

import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader

from models.flag.data.util import NodeType

from tggnn.model.basic import BasicModel

from ..common import Checkpoint
from .data import AdditiveNoiseTransform, Flag, FlagDataset
from .impl.model import flag_model
from .args import flag_model_args

# Prepare the argument parser
args = flag_model_args(True)

# Load the bubble trajectory dataset.
dataset = FlagDataset(
    args.data_directory,
    transforms=[AdditiveNoiseTransform(args.noise, 0.1)]
    if args.noise is not None
    else None,
)

# Set the torch seed
# torch.manual_seed(23)

# Setup the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = flag_model(
    dataset.layout(), args.latent_size, args.num_layers, args.iterations
).to(device)

optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=False)
scheduler = ExponentialLR(optimizer, gamma=(1e-2) ** (1 / 5e6))

flag_loader = DataLoader(
    dataset, batch_size=None, shuffle=True, num_workers=4, pin_memory=True
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
print("{:-<68}".format(""))
print("Epoch{:2}Train loss{:8}Time".format(" ", " ", " "))
print("{:-<68}".format(""))


# Train
def train(model: BasicModel[Flag], gather_statistics: bool) -> None:
    count = 0
    train_loss = 0
    epoch_start = cast(int, model.epoch.item())

    model.train()

    # Gather normalizer statistics
    if gather_statistics:
        for i, flag in enumerate(flag_loader):
            flag.to(device)

            model.gather_statistics(flag)

            if i >= 1000:
                break

    timer = default_timer()

    while True:
        for flag in flag_loader:
            flag.to(device)

            flag.node_type = flag.node_type.to(device)
            loss_mask = torch.squeeze(flag.node_type == NodeType.NORMAL)

            optimizer.zero_grad()

            out = model.forward(flag)

            pred = out.node_sets["flag"]["velocity"].attr
            label = out.labels["acceleration"]

            error = torch.sum((label - pred) ** 2, dim=1)
            loss = torch.mean(error[loss_mask])

            train_loss += float(loss)
            count += 1
            model.epoch += 1

            loss.backward()
            optimizer.step()
            scheduler.step()

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


train(model, gather_statistics)
