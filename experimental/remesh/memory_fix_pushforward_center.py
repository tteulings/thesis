from copy import deepcopy
import os
import sys
from timeit import default_timer

import numpy as np

import torch

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

from .bubble import Bubble, rotation_matrix_to_y_axis
from .sequence_dataset import BubbleSequenceDataset
# from .transforms import AdditiveNoiseTransform, RelativeNoiseTransform

import wandb

# from torchsummary import summary
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
    'center_predictions' : True,
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
    center_prediction = True,
    # ignore_sequences=[1, 2,3,4,5,6,7,8]
    start_point=2
)
# transform = AdditiveNoiseTransform(args.noise) if args.noise is not None else None

transform =None
# A small epsilon value to stabilize ARE computation.
eps = 1e-8

# Set the torch seed
# torch.manual_seed(23)

# Setup the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device= 'cpu'
model, state = bubble_memory_model(
    dataset.layout(), args.latent_size, args.num_layers, args.iterations
)

# print(summary(model))

# print(dataset.layout())

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

center_normalizer =  torch.tensor([2*10**8, .1*10**7, .25*10**5])

# center_normalizer =  torch.tensor([.7*10**7, .35*10**7, .5*10**5])

# center_normalizer =  torch.tensor([1, 1, 1])

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
                sequence, batch_size=None, shuffle=False, num_workers=2, pin_memory=True
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
            # centroid["memory"].attr.zero_()
            centroid["memory"].attr = torch.ones_like(centroid["memory"].attr)

            timer = default_timer()

            loss = torch.tensor([0.0], device=device)

            current = None
            ui = 0

            
            
            velocities = []

            velocities_ground_truth = []

            sequence.set_rotation_matrix(np.diag([1.0,1.0,1.0]))
            rotation_matrix = rotation_matrix_to_y_axis(sequence[0].center_velocity)

            rotation_matrix_groundtruth = rotation_matrix_to_y_axis(sequence[0].center_velocity)
            centroid['velocity'].attr = sequence[0].center_velocity
            sequence.set_rotation_matrix(rotation_matrix)

            mats = []
            # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
            for (count, bubble) in enumerate(sequence):
                print('\nstart')

                # print(count)

                if count == 0:
                    u = max(1, np.random.choice(list(range(int(args.tbptt*0.1), args.tbptt))   ))
                    ui = 0


                    if transform is not None:
                        current = transform(bubble)
                    else:
                        current = deepcopy(bubble)

                    # velocities_ground_truth.append(bubble.center_velocity)
                elif ui >  0 and (ui) ==  u:
                    print(count,'new_bubble')
                    u = max(1, np.random.choice(list(range(int(args.tbptt*0.1), args.tbptt))   ))
                    ui = 0

                    if transform is not None:
                        current = transform(bubble)
                    else:
                        current = deepcopy(bubble)
                    print(current.center_velocity)
                else:
                    if current is None:
                        # Just to satisfy the type system.
                        # This should never happen.
                        return

                    current.remesh(bubble.instructions)
                    current.labels["target"] = bubble.labels["target"]

                    current.center_target = bubble.center_target
                    
                print(ui, u, count)


                velocities.append((current.center_velocity).cpu().numpy())

                current.to(device)

   
                print('value', centroid['velocity'].attr, current.center_velocity.to(device) )
                state = center_memory(current, state)

                # centroid['velocity'].attr = bubble.center_target.to(device).reshape((1,3))*center_normalizer.to(device)
                out, state, center_out = model.forward(current, state)

                wandb.log({'centroid': centroid["memory"].attr.cpu().detach().numpy()[0]})

                pred = out.node_sets["bubble"]["velocity"].attr
                label = out.labels["target"]

                error = ((bubble.center_target.to(center_out.device)*center_normalizer.to(center_out.device)-center_out.reshape(3))**2)
                
                # current.update(
                #     model._label_normalizers["target"].inverse(label), False
                # )

                one_step_loss = F.mse_loss(pred, label)
                one_step_center_loss = F.mse_loss(bubble.center_target.to(center_out.device)*center_normalizer.to(center_out.device), center_out.reshape(3))

                # loss += one_step_center_loss  + one_step_loss
                loss += one_step_center_loss

                train_loss += float(one_step_center_loss.item())

                model.epoch += 1
                # print(bubble.center_target.to(center_out.device)*center_normalizer.to(center_out.device))
            


                wandb.log({'u': u, 'loss_x': error[0], 'loss_y': error[1], 'loss_z': error[2], 'epoch': model.epoch, 'bubble_num': si+1, "loss": loss, 'onesteploss': one_step_loss , 'onestepcenterloss': one_step_center_loss})

                # print(count,args.tbptt, count%args.tbptt)
                # TBPTT with k1 == k2
                if ui == u-1:
                    
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    # scheduler.step()
                    print(count, 'backprop\n')



                if (u - ui) >= 5  or  ui == u-1:
                    print('detach')
                    # TODO: I would rather use the inplace `detach_()`. However, I
                    # have a feeling that it doesn't have the same effect on the
                    # autograd graph. Find out if there are any differences.
                    centroid["memory"].attr = centroid["memory"].attr.detach()
                    loss.detach_().zero_()
                    error.detach_().zero_()

                    one_step_loss.detach_().zero_()
                    one_step_center_loss.detach_().zero_()
                    centroid['velocity'].attr.detach_()
                else:
                    print(count, 'not detach')



                if count > -1:

                    velocities_ground_truth.append(bubble.center_target.cpu().numpy())
                    print(velocities_ground_truth)

                    

                    # print(np.sum(velocities[max(0, count-10):count], axis=0))
                    # print(len(velocities), velocities[max(0, len(velocities)-10):count])
                    # rotation_matrix = rotation_matrix_to_y_axis(( np.sum(velocities[max(0, len(velocities)-10):count+1], axis=0)) )
                    # rotation_matrix_groundtruth = rotation_matrix_to_y_axis(( np.sum(velocities_ground_truth[max(0, len(velocities_ground_truth)-10):count+1], axis=0)) )
                    rotation_matrix = rotation_matrix_to_y_axis(np.sum(velocities[-1:], axis=0))
                    velocities = list(velocities@rotation_matrix.T)
                    rotation_matrix_groundtruth = rotation_matrix_to_y_axis(np.sum(velocities_ground_truth[-1:], axis=0))@rotation_matrix_groundtruth
                    sequence.set_rotation_matrix(rotation_matrix_groundtruth)

                    # print(bubble.rotation_matrix)
                    # if len(velocities) >= 10:
                    #     velocities.pop(0)
                    #     velocities_ground_truth.pop(0)
                    # velocities = [max(0, count-10)]
                    # print(velocities)
                    # rotation_matrix = rotation_matrix_to_y_axis(().cpu().numpy() )
                    # sequence.set_rotation_matrix(rotation_matrix)



                vel_pred = bubble.center_target.reshape((3))


                print('center_target', vel_pred)
                # vel_true
                current_centroid = current.centroid(True)

                current.update(
                    model._label_normalizers["target"].inverse(pred.detach()), False, center_out / center_normalizer.to(center_out.device),rotation_matrix
                )
                centroid['velocity'].attr = center_out / center_normalizer.to(center_out.device)

                # current.update(
                #     model._label_normalizers["target"].inverse(pred.detach()), False, current.center_target
                # )
                # vel_pred = current.centroid() - current_centroid
                # vel_pred = bubble.center_target.reshape((3))
                
                vel_center_pred = (center_out / center_normalizer.to(center_out.device))[0]
                # vel_pred = current.center_target

                bubble_centroid = bubble.centroid()
      

                bubble.update(
                    model._label_normalizers["target"].inverse(label).to(bubble.device), False, bubble.center_target
                )



    
                vel_true = bubble.centroid() - bubble_centroid

                wandb.log({'u': u, 'loss_x': error[0], 'loss_y': error[1], 'loss_z': error[2], 'epoch': model.epoch, 'bubble_num': si+1,'velocity_centroid_x2': vel_center_pred[ 0], 'velocity_centroid_y2': vel_center_pred[1], 'velocity_centroid_z2': vel_center_pred[ 2],  'velocity_centroid_x': vel_pred[ 0], 'velocity_centroid_y': vel_pred[1], 'velocity_centroid_z': vel_pred[ 2], 'true_velocity_centroid_x': vel_true[ 0], 'true_velocity_centroid_y': vel_true[1], 'true_velocity_centroid_z': vel_true[ 2]})
                ui+=1
                if count > 500:
                    break






                if (count + 1) % 500 == 0:
              
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
