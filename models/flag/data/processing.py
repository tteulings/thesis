from typing import Optional

import functools
import io
import json
import os
import sqlite3

import tensorflow as tf

import torch
from torch import Tensor

from tggnn.data.index import SortedIndex
from tggnn.data.typed_graph import TypedGraphLayout, TypedGraphProcessor
from tggnn.data.util import face_to_edge

from ...common import windowed
from .flag import Flag
from .util import parse_tf_record


def tensor_to_binary(tensor: Tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)

    buffer.seek(0)
    return buffer.read()


class FlagProcessor(TypedGraphProcessor):
    def __init__(self, root: str, record_name: str) -> None:
        super().__init__(root)

        self.layout: Optional[TypedGraphLayout] = None
        self.record_name = record_name

        with open(os.path.join(self.root, "meta.json")) as meta_file:
            self.meta = json.load(meta_file)

        db_path = os.path.join(self.root, "flag_static.db")

        if os.path.exists(db_path):
            os.remove(db_path)

        self._db_con = sqlite3.connect(db_path)

        self._db_con.execute(
            'CREATE TABLE "mesh" ('
            '"id" INTEGER NOT NULL UNIQUE,'
            '"index" BLOB NOT NULL,'
            '"faces" BLOB NOT NULL,'
            '"position" BLOB NOT NULL,'
            '"node_type" BLOB NOT NULL,'
            'PRIMARY KEY("id" AUTOINCREMENT)'
            ");"
        )
        self._db_con.execute(
            'CREATE TABLE "position" ('
            '"id"	INTEGER NOT NULL UNIQUE,'
            '"value" BLOB NOT NULL,'
            'PRIMARY KEY("id" AUTOINCREMENT)'
            ");"
        )
        self._db_con.execute(
            'CREATE TABLE "sequence" ('
            '"id" INTEGER NOT NULL UNIQUE,'
            '"mesh_id" INTEGER NOT NULL,'
            '"length" INTEGER NOT NULL,'
            'FOREIGN KEY("mesh_id") REFERENCES "mesh"("id"),'
            'PRIMARY KEY("id" AUTOINCREMENT)'
            ");"
        )
        self._db_con.execute(
            'CREATE TABLE "flag" ('
            '"id" INTEGER NOT NULL UNIQUE,'
            '"sequence_id" INTEGER NOT NULL,'
            '"prev" INTEGER NOT NULL,'
            '"cur" INTEGER NOT NULL,'
            '"next" INTEGER NOT NULL,'
            'FOREIGN KEY("next") REFERENCES "position"("id"),'
            'FOREIGN KEY("cur") REFERENCES "position"("id"),'
            'FOREIGN KEY("sequence_id") REFERENCES "sequence"("id"),'
            'FOREIGN KEY("prev") REFERENCES "position"("id"),'
            'PRIMARY KEY("id" AUTOINCREMENT)'
            ");"
        )

    def __process__(self) -> TypedGraphLayout:
        ds = tf.data.TFRecordDataset(os.path.join(self.root, self.record_name))
        ds = ds.map(
            functools.partial(parse_tf_record, meta=self.meta),
            num_parallel_calls=1,
        )

        cursor = self._db_con.cursor()
        cursor.execute("begin")

        first = True

        for sequence_id, record in enumerate(ds):
            world_pos = torch.tensor(
                record["world_pos"].numpy(), dtype=torch.float32
            )

            if first:
                first = False

                node_type = torch.tensor(
                    record["node_type"][0].numpy(), dtype=torch.long
                )

                cells = torch.tensor(
                    record["cells"][0].numpy().transpose(), dtype=torch.long
                )

                mesh_index = face_to_edge(cells)

                # NOTE: Remove all edges going to HANDLE nodes.
                # mask = node_type[mesh_index[1, :]] != NodeType.HANDLE
                # mesh_index = mesh_index[:, mask.squeeze()]

                mesh_pos = torch.tensor(
                    record["mesh_pos"][0].numpy(), dtype=torch.float32
                )

                cursor.execute(
                    "INSERT INTO mesh ('index', faces, position, node_type)"
                    " VALUES (?,?,?,?)",
                    [
                        tensor_to_binary(mesh_index),
                        tensor_to_binary(cells),
                        tensor_to_binary(mesh_pos),
                        tensor_to_binary(node_type),
                    ],
                )

                # Instantiate one flag to get the layout.
                self.layout = Flag(
                    [world_pos[0], world_pos[1], world_pos[2]],
                    cells,
                    SortedIndex(mesh_index),
                    mesh_pos,
                    node_type,
                ).layout()

            cursor.execute(
                "INSERT INTO sequence (mesh_id, length) VALUES (1, ?)",
                [world_pos.size()[0] - 2],
            )

            cursor.execute("SELECT MAX(id) FROM position")
            record = cursor.fetchone()

            from_id = 1 if record[0] is None else record[0]

            for pos_index in range(world_pos.size()[0]):
                cursor.execute(
                    "INSERT INTO position (value) VALUES (?)",
                    [tensor_to_binary(world_pos[pos_index].clone())],
                )

            for positions in windowed(
                range(from_id, from_id + world_pos.size()[0]), 3
            ):
                cursor.execute(
                    "INSERT INTO flag (sequence_id, prev, cur, next) VALUES (?,"
                    " ?, ?, ?)",
                    [sequence_id + 1, *positions],
                )

        cursor.execute("commit")

        if self.layout is None:
            raise Exception(
                "Could not obtain TypedGraphLayout when processing."
            )

        return self.layout
