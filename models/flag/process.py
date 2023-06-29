import argparse

from .data import FlagProcessor

from timeit import default_timer

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_directory", help="Path of the data directory to load."
)
parser.add_argument("record_name", help="Name of the record file to load.")

args = parser.parse_args()

t = default_timer()
FlagProcessor(args.data_directory, args.record_name).process(True)
print(default_timer() - t)
