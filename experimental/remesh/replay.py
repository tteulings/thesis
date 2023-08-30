import ctypes
from numpy.ctypeslib import ndpointer

import numpy as np
from numpy.typing import NDArray

from models.bubble.data.config import SimulationConfig

_lib = ctypes.cdll.LoadLibrary(
    "./models/bubble/extern/remesh-cpp/build/libremesher.so"
)


class _ReplayOutput(ctypes.Structure):
    _fields_ = [
        ("num_nodes", ctypes.c_int32),
        ("num_cells", ctypes.c_int32),
        ("positions", ctypes.POINTER(ctypes.c_double)),
        ("markpos", ctypes.POINTER(ctypes.c_int32)),
        ("connect", ctypes.POINTER(ctypes.c_int32)),
    ]

_free_remesh_output = _lib.free_replay_output
_free_remesh_output.argtypes = [_ReplayOutput]

class ReplayResult:
    def __init__(
        self,
        values: NDArray[np.float64],
        faces: NDArray[np.int32],
        connect: NDArray[np.int32],
    ):
        self.values = values
        self.faces = faces
        self.connect = connect


_replay = _lib.replay
_replay.restype = _ReplayOutput
_replay.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_ubyte, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_bool,
]


def replay(
    config: SimulationConfig,
    init_vol: float,
    values: NDArray[np.float64],
    faces: NDArray[np.int32],
    connect: NDArray[np.int32],
    instructions: NDArray[np.ubyte],
    remesh_velocities: bool,
) -> ReplayResult:

    # print(values)
    bubble = _replay(
        config.dx,
        config.dy,
        config.dz,
        init_vol,
        config.fak_min,
        config.fak_max,
        np.ascontiguousarray(values, dtype=float),
        len(values),
        np.ascontiguousarray(faces),
        np.ascontiguousarray(connect),
        len(faces),
        instructions,
        len(instructions),
        remesh_velocities,
    )

    ret_pos = np.ctypeslib.as_array(
        bubble.positions, shape=(bubble.num_nodes, 3)
    )
    ret_faces = np.ctypeslib.as_array(
        bubble.markpos, shape=(bubble.num_cells, 3)
    )
    ret_connect = np.ctypeslib.as_array(
        bubble.connect, shape=(bubble.num_cells, 3)
    )

    
    ret_pos = np.copy(np.ctypeslib.as_array(bubble.positions, shape=(bubble.num_nodes, 3)))
    ret_faces = np.copy(np.ctypeslib.as_array(bubble.markpos, shape=(bubble.num_cells, 3)))
    ret_connect = np.copy(np.ctypeslib.as_array(bubble.connect, shape=(bubble.num_cells, 3)))

    _free_remesh_output(bubble)


    return ReplayResult(ret_pos, ret_faces, ret_connect)
