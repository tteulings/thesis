from collections import namedtuple
from typing import List

import numpy as np

Bubble = namedtuple("Bubble", ["positions", "velocities", "faces", "connect"])


def read_int(fid):
    # return int.from_bytes(fid.read(4), "little", signed=True)
    return np.frombuffer(fid.read(4), dtype=np.int32)[0]


def read_double(fid):
    return np.frombuffer(fid.read(8), dtype=np.float64)[0]


class BinFile:
    def __init__(self, fname: str):
        self.fname = fname
        self.origin: np.ndarray
        self.Bubbles: List[Bubble] = []

    def load(self) -> "BinFile":
        fid = open(self.fname, "rb")
        cycle = read_int(fid)

        # print("Reading bin file: cycle number", cycle)

        num_bubbles = read_int(fid)
        has_velocities = read_int(fid)

        if not has_velocities:
            print(
                f"[WARNING] Data for cycle {cycle} misses velocity data,"
                " defaulting to zeros."
            )

        self.origin = np.fromfile(fid, dtype=np.float64, count=3)

        for _ in range(num_bubbles):
            nmar = read_int(fid)
            npos = read_int(fid)

            positions = np.reshape(
                np.fromfile(fid, dtype=np.float64, count=npos * 3), (npos, 3)
            )

            if has_velocities:
                velocities = np.reshape(
                    np.fromfile(fid, dtype=np.float64, count=npos * 3),
                    (npos, 3),
                )
            else:
                velocities = np.zeros((npos, 3))

            faces = np.reshape(
                np.fromfile(fid, dtype=np.int32, count=nmar * 3), (nmar, 3)
            )

            connect = np.reshape(
                np.fromfile(fid, dtype=np.int32, count=nmar * 3), (nmar, 3)
            )

            self.Bubbles.append(Bubble(positions, velocities, faces, connect))

        return self

    def volume(self, bubble=0):
        pos = self.Bubbles[bubble].positions
        faces = self.Bubbles[bubble].faces

        return (
            np.sum(
                pos[faces[:, 0]]
                * np.cross(pos[faces[:, 1]], pos[faces[:, 2]], axis=1)
            )
            / 6.0
        )
