import gcl
import torch
from sequence_dataset import BubbleSequenceDataset

# Load the bubble trajectory dataset.
dataset = BubbleSequenceDataset(
    './',
    # transforms=[RelativeNoiseTransform(args.noise)]
    # if args.noise is not None
    # else None,
    remesh_velocity=True,
    target_acceleration=False,
    # ignore_sequences=[1, 2,3,4,5,6,7,8]
    start_point=2
)