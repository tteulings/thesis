from typing import List, Optional, cast

import io
from os import path
import jsonpickle
import sqlite3

import torch
from torch.utils.data import Dataset

from tggnn.data.index import SortedIndex
from tggnn.data.typed_graph import TypedGraphLayout, TypedGraphTransform

from .flag import Flag


class FlagDataset(Dataset[Flag]):
    def __init__(
        self,
        root: str,
        transforms: Optional[List[TypedGraphTransform[Flag]]] = None,
    ):
        super().__init__()

        self._root = root
        self._transforms = transforms

        with open(path.join(root, "layout.json"), "r") as file:
            self._layout = cast(
                TypedGraphLayout, jsonpickle.decode(file.read())
            )

        self._db = sqlite3.connect(path.join(self._root, "flag_static.db"))
        self._cursor = self._db.cursor()

    def __len__(self) -> int:
        return self._cursor.execute(
            "SELECT SUM(length) FROM sequence"
        ).fetchone()[0]

    def __getitem__(self, idx: int) -> Flag:
        self._cursor.execute(
            "SELECT mesh.'index' as 'index', mesh.faces as faces, "
            "mesh.position as mesh_pos, mesh.node_type as node_type, "
            "prev.value as prev, cur.value as cur, next.value as next "
            "FROM flag "
            "LEFT JOIN position prev "
            "ON flag.prev = prev.id "
            "LEFT JOIN position cur "
            "ON flag.cur = cur.id "
            "LEFT JOIN position next "
            "ON flag.next = next.id "
            "LEFT JOIN sequence "
            "ON flag.sequence_id = sequence.id "
            "LEFT JOIN mesh "
            "ON sequence.mesh_id = mesh.id "
            "WHERE flag.id = ?",
            [idx + 1],
        )
        result = self._cursor.fetchone()

        index, faces, mesh_pos, node_type, prev, cur, next = (
            torch.load(io.BytesIO(bytes)) for bytes in result
        )

        return Flag(
            [prev, cur, next], faces, SortedIndex(index), mesh_pos, node_type
        )

    def layout(self) -> TypedGraphLayout:
        return self._layout
