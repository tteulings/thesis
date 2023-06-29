import argparse

from .data import BubbleProcessor
from .data.config import SimulationConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_directory", help="Name of the data directory to load."
)
parser.add_argument(
    "-n",
    "--mesh-size",
    choices=[10, 50],
    type=int,
    help="The mesh size in each dimension",
)
# parser.add_argument(
#     "-w",
#     "--num-workers",
#     default=None,
#     type=int,
#     help="The number of worker threads to use when processing the data.",
# )
args = parser.parse_args()

# NOTE: Configs for n=10, and n=50, respectively.
sim_configs = {
    10: SimulationConfig(1e-3, 1e-3, 1e-3, 1e-4, 2e-1, 5e-1),
    50: SimulationConfig(4e-4, 4e-4, 4e-4, 1e-4, 2e-1, 5e-1),
}

# config = BubbleProcessConfig(
#     sim_configs[args.mesh_size], num_workers=args.num_workers
# )

BubbleProcessor(args.data_directory, sim_configs[args.mesh_size]).process(True)
