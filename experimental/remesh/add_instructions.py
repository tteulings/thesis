import io
import sqlite3
from os import path
from typing import cast

import jsonpickle
import numpy as np
from argparse import ArgumentParser

from models.bubble.data import SimulationConfig, queries
from models.bubble.data.processing import ndarray_to_binary

from .bubble import bubble_volume
from .remesh import remesh

parser = ArgumentParser()
parser.add_argument("data_directory", help="The directory to load data from.")
args = parser.parse_args()

with open(path.join(args.data_directory, "config.json")) as config_file:
    config = cast(SimulationConfig, jsonpickle.decode(config_file.read()))

db = sqlite3.connect(path.join(args.data_directory, "bubble.db"))

db.execute('ALTER TABLE bubble ADD instructions BLOB NOT NULL DEFAULT ""')

length = db.execute("SELECT SUM(length) FROM sequence").fetchone()[0]

cursor = db.cursor()

cursor.execute("BEGIN")

for idx in range(length):
    print(idx)

    faces, connect, prev, cur, next = (
        np.load(io.BytesIO(bytes))
        for bytes in db.execute(
            queries.select_bubble_by_id, [idx + 1]
        ).fetchone()
    )

    remesh_result = remesh(
        config, bubble_volume(prev, faces).item(), cur, faces, connect
    )

    db.execute(
        "UPDATE bubble SET instructions = ? WHERE id = ?",
        [ndarray_to_binary(remesh_result.instructions), idx],
    )

    if idx % 8000 == 0:
        cursor.execute("COMMIT")
        cursor.execute("BEGIN")

cursor.execute("COMMIT")
