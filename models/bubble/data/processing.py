from typing import cast, Optional

import io
import os
import sqlite3
from timeit import default_timer

import jsonpickle
import numpy as np
from numpy import ndarray

from tggnn.data.typed_graph import TypedGraphLayout, TypedGraphProcessor

from . import queries
from .bin import BinFile
from .bubble import Bubble
from .config import SimulationConfig


def ndarray_to_binary(array: ndarray):
    buffer = io.BytesIO()
    np.save(buffer, array)

    buffer.seek(0)
    return buffer.read()


class BubbleProcessor(TypedGraphProcessor):
    def __init__(self, root: str, config: SimulationConfig) -> None:
        super().__init__(root)

        self.config = config
        self.layout: Optional[TypedGraphLayout] = None

        self.db_path = os.path.join(self.root, "bubble.db")

    def __process__(self) -> TypedGraphLayout:
        t = default_timer()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        with open(os.path.join(self.root, "config.json"), "w") as config_file:
            config_file.write(cast(str, jsonpickle.encode(self.config)))

        _db_con = sqlite3.connect(self.db_path)

        _db_con.execute(queries.create_sequence_table)
        _db_con.execute(queries.create_mesh_table)
        _db_con.execute(queries.create_geometry_table)
        _db_con.execute(queries.create_bubble_table)

        for dir in filter(
            lambda item: os.path.isdir(item),
            (os.path.join(self.root, item) for item in os.listdir(self.root)),
        ):
            print(f"Processing data in {dir}.")

            cursor = _db_con.cursor()
            cursor.execute("begin")

            # NOTE: Discard the first 0.2 seconds (Roghair et. al 2016, par 2.7,
            # fig 6b).
            # Define preliminary sequential samplers.
            prev_names = (
                os.path.join(dir, f"F{i}_post.bin")
                for i in range(19990, 99990, 10)
            )
            data_names = (
                os.path.join(dir, f"F{i}_pre.bin")
                for i in range(20000, 100000, 10)
            )
            label_names = (
                os.path.join(dir, f"F{i}_pre.bin")
                for i in range(20010, 100010, 10)
            )

            cursor.execute("INSERT INTO sequence (length) VALUES (8000)")
            sequence_id = cursor.lastrowid

            first = True

            for file_triplet in zip(prev_names, data_names, label_names):
                prev, current, next = (
                    BinFile(file).load() for file in file_triplet
                )

                if first:
                    cursor.execute(
                        "INSERT INTO mesh (faces, connect) VALUES (?, ?)",
                        [
                            ndarray_to_binary(
                                prev.Bubbles[0].faces.astype(np.int32)
                            ),
                            ndarray_to_binary(
                                prev.Bubbles[0].connect.astype(np.int32)
                            ),
                        ],
                    )
                    source_mesh_id = cursor.lastrowid

                    cursor.execute(
                        "INSERT INTO geometry (mesh_id, positions) VALUES"
                        " (?, ?)",
                        [
                            source_mesh_id,
                            ndarray_to_binary(
                                (
                                    current.origin
                                    + current.Bubbles[0].positions
                                ).astype(np.float64),
                            ),
                        ],
                    )
                    current_geom_id = cursor.lastrowid

                    first = False
                else:
                    cursor.execute("SELECT MAX(id) FROM mesh")
                    source_mesh_id = cursor.fetchone()[0]

                    cursor.execute("SELECT MAX(id) FROM geometry")
                    current_geom_id = cursor.fetchone()[0]

                # NOTE: prev is defined on the same mesh as current.
                cursor.execute(
                    "INSERT INTO geometry (mesh_id, positions) VALUES (?, ?)",
                    [
                        source_mesh_id,
                        ndarray_to_binary(
                            (prev.origin + prev.Bubbles[0].positions).astype(
                                np.float64
                            ),
                        ),
                    ],
                )
                prev_geom_id = cursor.lastrowid

                cursor.execute(
                    "INSERT INTO mesh (faces, connect) VALUES (?, ?)",
                    [
                        ndarray_to_binary(
                            next.Bubbles[0].faces.astype(np.int32)
                        ),
                        ndarray_to_binary(
                            next.Bubbles[0].connect.astype(np.int32)
                        ),
                    ],
                )
                target_mesh_id = cursor.lastrowid

                cursor.execute(
                    "INSERT INTO geometry (mesh_id, positions) VALUES (?, ?)",
                    [
                        target_mesh_id,
                        ndarray_to_binary(
                            (next.origin + next.Bubbles[0].positions).astype(
                                np.float64
                            )
                        ),
                    ],
                )
                next_geom_id = cursor.lastrowid

                cursor.execute(
                    "INSERT INTO bubble (sequence_id, previous, current, next)"
                    " VALUES (?, ?, ?, ?)",
                    [sequence_id, prev_geom_id, current_geom_id, next_geom_id],
                )

            cursor.execute("commit")

        print(f"Processing time: {default_timer() - t}.")

        cursor = _db_con.execute(queries.select_bubble_by_id, [1])
        result = cursor.fetchone()

        faces, connect, prev, cur, next = (
            np.load(io.BytesIO(bytes)) for bytes in result
        )

        return Bubble(self.config, faces, connect, [prev, cur, next]).layout()
