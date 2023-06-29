from typing import cast, List, Optional

import io
import sqlite3
from os import path
from sqlite3.dbapi2 import Connection

import jsonpickle
import numpy as np

from torch.utils.data.dataset import Dataset

from tggnn.data.typed_graph import (
    TypedGraphDataset,
    TypedGraphLayout,
    TypedGraphTransform,
)

from models.bubble.data import SimulationConfig, queries

from .bubble import Bubble


class BubbleSequence(TypedGraphDataset[Bubble]):
    _config: SimulationConfig
    _db: Connection
    _sequence_id: int
    _base_id: int

    def __init__(
        self,
        root: str,
        sequence_id: int,
        transforms: Optional[List[TypedGraphTransform[Bubble]]] = None,
        remesh_velocity: bool = False,
        target_acceleration: bool = False,
    ) -> None:
        super().__init__(root, transforms)

        with open(path.join(root, "config.json")) as config_file:
            self._config = cast(
                SimulationConfig, jsonpickle.decode(config_file.read())
            )

        self._db = sqlite3.connect(path.join(root, "bubble.db"))

        self._sequence_id = sequence_id

        base_id = self._db.execute(
            "SELECT SUM(length) FROM sequence WHERE id < ?", [self._sequence_id]
        ).fetchone()[0]

        self._base_id = 0 if base_id is None else base_id

        self._remesh_velocity = remesh_velocity
        self._target_acceleration = target_acceleration

    def __len__(self) -> int:
        print(self._sequence_id)
        return self._db.execute(
            "SELECT length FROM sequence WHERE id = ?", [self._sequence_id]
        ).fetchone()[0]

    def __get__(self, idx: int) -> Bubble:
        faces, connect, prev, cur, next = (
            np.load(io.BytesIO(bytes))
            for bytes in self._db.execute(
                queries.select_bubble_by_id, [self._base_id + idx + 1]
            ).fetchone()
        )
        return Bubble(
            self._config,
            faces,
            connect,
            [prev, cur, next],
            self._remesh_velocity,
            self._target_acceleration,
        )


class BubbleSequenceDataset(Dataset[BubbleSequence]):
    def __init__(
        self,
        root: str,
        transforms: Optional[List[TypedGraphTransform[Bubble]]] = None,
        remesh_velocity: bool = False,
        target_acceleration: bool = False,
        ignore_sequences: List[int] = [],
        sequences: List[int] = [],
        start_point: int = 1

    ) -> None:
        super().__init__()

        self._root = root
        self._transforms = transforms

        with open(path.join(root, "layout.json"), "r") as file:
            self._layout = cast(
                TypedGraphLayout, jsonpickle.decode(file.read())
            )

        self._db = sqlite3.connect(path.join(root, "bubble.db"))

        self._remesh_velocity = remesh_velocity
        self._target_acceleration = target_acceleration

    
        self._ignore_sequences = ignore_sequences

        if len(ignore_sequences) > 0:
            self._sequences = [int(i) for i in np.arange(start_point, self._db.execute("SELECT MAX(id) FROM sequence").fetchone()[0]+1) if i not in ignore_sequences] 
        elif len(sequences) > 0:
            self._sequences = sequences
        else:
            self._sequences = [int(i) for i in np.arange(1, self._db.execute("SELECT MAX(id) FROM sequence").fetchone()[0]+1)] 
    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> BubbleSequence:
        print(self._sequences[index])
        return BubbleSequence(
            self._root,
            self._sequences[index],
            self._transforms,
            self._remesh_velocity,
            self._target_acceleration,
        )

    def layout(self) -> TypedGraphLayout:
        return self._layout
