import ctypes
from numpy.ctypeslib import ndpointer

import numpy as np
from numpy.typing import NDArray

from models.bubble.data.config import SimulationConfig

_lib = ctypes.cdll.LoadLibrary(
    "./models/bubble/extern/remesh-cpp/build/libremesher.so"
)


class _RemeshOutput(ctypes.Structure):
    _fields_ = [
        ("num_nodes", ctypes.c_int32),
        ("num_cells", ctypes.c_int32),
        ("num_instr", ctypes.c_int32),
        ("positions", ctypes.POINTER(ctypes.c_double)),
        ("markpos", ctypes.POINTER(ctypes.c_int32)),
        ("connect", ctypes.POINTER(ctypes.c_int32)),
        ("instructions", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class RemeshResult:
    def __init__(
        self,
        positions: NDArray[np.float64],
        faces: NDArray[np.int32],
        connect: NDArray[np.int32],
        instructions: NDArray[np.ubyte],
    ):
        self.positions = positions
        self.faces = faces
        self.connect = connect
        self.instructions = instructions


_remesh = _lib.remesh
_remesh.restype = _RemeshOutput
_remesh.argtypes = [
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
]


def remesh(
    config: SimulationConfig,
    init_vol: float,
    positions: NDArray[np.float64],
    faces: NDArray[np.int32],
    connect: NDArray[np.int32],
) -> RemeshResult:
    bubble = _remesh(
        config.dx,
        config.dy,
        config.dz,
        init_vol,
        config.fak_min,
        config.fak_max,
        positions,
        len(positions),
        faces,
        connect,
        len(faces),
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

    # 16 bytes per instruction
    instructions = np.ctypeslib.as_array(
        bubble.instructions, shape=(bubble.num_instr, 16)
    )

    return RemeshResult(ret_pos, ret_faces, ret_connect, instructions)
