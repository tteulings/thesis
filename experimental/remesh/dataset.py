from typing import List, Optional, cast

import io
import sqlite3
from os import path

import jsonpickle
import numpy as np

from tggnn.data.typed_graph import TypedGraphDataset, TypedGraphTransform

# from models.bubble.data.bubble import Bubble
from models.bubble.data import queries, SimulationConfig

from .bubble import Bubble


class BubbleDataset(TypedGraphDataset[Bubble]):
    def __init__(
        self,
        root: str,
        transforms: Optional[List[TypedGraphTransform[Bubble]]] = None,
        remesh_velocity: bool = False,
        target_acceleration: bool = False,
        center_prediction: bool = False,
        rotation_matrix: np.array = []

    ) -> None:
        super().__init__(root, transforms)

        with open(path.join(root, "config.json")) as config_file:
            self._config = cast(
                SimulationConfig, jsonpickle.decode(config_file.read())
            )

        self._db = sqlite3.connect(path.join(root, "bubble.db"))

        self._remesh_velocity = remesh_velocity
        self._target_acceleration = target_acceleration
        self._center_prediction = center_prediction

        self._rotation_matrix = np.diag([1.0,1.0,1.0])


    def set_rotation_matrix(self, rotation_matrix):
        # print('set_matrix',rotation_matrix)
        self._rotation_matrix = rotation_matrix
    def __len__(self) -> int:
        return self._db.execute("SELECT SUM(length) FROM sequence").fetchone()[
            0
        ]

    def __get__(self, idx: int) -> Bubble:
        faces, connect, prev, cur, next = (
            np.load(io.BytesIO(bytes))
            for bytes in self._db.execute(
                queries.select_bubble_by_id, [idx + 1]
            ).fetchone()
        )
        # print('get')

        return Bubble(
            self._config,
            faces,
            connect,
            [prev, cur, next],
            self._remesh_velocity,
            self._target_acceleration,
            center_prediction=self._center_prediction,
            rotation_matrix=self._rotation_matrix
        )
 
