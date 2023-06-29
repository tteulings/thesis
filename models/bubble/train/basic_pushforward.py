import os
import sys
from copy import deepcopy
from timeit import default_timer

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from tggnn.model.basic import BasicModel
from tggnn.network.loss import point_to_surface_loss

from ...common import Checkpoint, windowed
from ..args import bubble_model_args
from ..data import Bubble, BubbleSequenceDataset
from ..data.transforms import AdditiveNoiseTransform
from ..impl.basic import bubble_basic_model

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
print(
    "Epoch{:3}MSE loss{:9}Adversarial loss{:3}Time".format(" ", " ", " ")
    if args.pushforward
    else "Epoch{:3}MSE loss{:9}Time".format(" ", " ")
)
print("{:-<80}".format(""))

# Gather normalizer statistics
model.train()

# Run model epochs
def train(model: BasicModel[Bubble], gather_statistics: bool) -> None:
    count = 0
    train_loss = 0
    total_adversarial_loss = 0

    epoch_start = model.epoch.item()

    while True:
        for sequence in sequence_loader:
            bubble_loader = DataLoader(
                sequence,
                batch_size=None,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            if gather_statistics:
                for i, bubble in enumerate(bubble_loader):
                    bubble.to(device)

                    model.gather_statistics(bubble)

                    if i >= 1000:
                        break

            timer = default_timer()

            for bubble, _, next in windowed(iter(bubble_loader), 3):
                bubble.to(device)
                bubble.positions = bubble.positions.to(device)
                # next.to(device)

                optimizer.zero_grad()

                out = model.forward(deepcopy(bubble))

                pred = out.node_sets["bubble"]["velocity"].attr
                label = out.labels["target"]

                # FIXME: This only works with velocity targets.
                # one_step_loss = point_to_surface_loss(
                #     bubble.positions.float()
                #     + model._label_normalizers["target"].inverse(pred),
                #     target.positions.float().to(device),
                #     target.faces.to(device),
                # )

                one_step_loss = F.mse_loss(pred, label)

                train_loss += float(one_step_loss)
                count += 1
                model.epoch += 1

                loss = one_step_loss

                if args.pushforward:
                    bubble.update(
                        model._label_normalizers["target"].inverse(pred)
                    )
                    out = model.forward(deepcopy(bubble))

                    pred = out.node_sets["bubble"]["velocity"].attr

                    # FIXME: This only works with velocity targets.
                    adversarial_loss = point_to_surface_loss(
                        bubble.positions.float()
                        + model._label_normalizers["target"].inverse(pred),
                        next.positions.float().to(device),
                        next.faces.to(device),
                    )

                    loss += adversarial_loss
                    total_adversarial_loss += float(adversarial_loss)

                loss.backward()
                optimizer.step()
                # scheduler.step()

                if count % args.print_every == 0:
                    print(
                        f"{model.epoch.item():<8}{train_loss/count:<17.8e}",
                        end="",
                    )
                    if args.pushforward:
                        print(f"{total_adversarial_loss/count:<19.8e}", end="")

                    print(f"{default_timer() - timer}")
                    sys.stdout.flush()

                    train_loss = 0
                    total_adversarial_loss = 0
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
