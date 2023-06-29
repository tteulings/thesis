from collections import namedtuple

import numpy as np

Bubble = namedtuple(
    "Bubble", ["positions", "connectivity", "points", "velocity"]
)


def ft3int(fid):
    return int.from_bytes(fid.read(4), "little", signed=True)


def ft3double(fid):
    # return float.from_bytes(fid.read(8), "little", signed=True)
    return np.frombuffer(fid.read(8), dtype=float)[0]


def ft3skipbytes(fid, n):
    return fid.read(n)


class Ft3File:
    def __init__(self, fname):
        # Set filename
        self.fname = fname
        self.Bubbles = []

    def load(self):
        # Read through the binary data file format, and
        # store relevant parameters to extract bubble meshes
        fid = open(self.fname, "rb")
        cycle = ft3int(fid)
        print("Reading FT3 file: Cycle number", cycle)

        # Skip bytes: 4 (dummy int) + 4*8 (time and originshift x,y,z)
        # ft3skipbytes(fid, 36)  # Dummy
        ft3skipbytes(fid, 12)
        self.xshift = ft3double(fid)
        self.yshift = ft3double(fid)
        self.zshift = ft3double(fid)
        # print("({}, {}, {})".format(self.xshift, self.yshift, self.zshift))

        # Read grid dimensions:
        self.nx = ft3int(fid)
        ft3skipbytes(fid, 4)  # Dummy
        self.ny = ft3int(fid)
        ft3skipbytes(fid, 4)  # Dummy
        self.nz = ft3int(fid)
        ft3skipbytes(fid, 4)  # Dummy

        # Skip bytes: 3*8: dx, dy, dz (doubles)
        # fid.read(24)
        self.dx = ft3double(fid)
        self.dy = ft3double(fid)
        self.dz = ft3double(fid)
        # print("({}, {}, {})".format(self.dx, self.dy, self.dz))
        self.nph = ft3int(fid)
        ft3skipbytes(fid, 4)  # Dummy
        self.neli = ft3int(fid)
        has_velocities = ft3int(fid)

        # Skip bytes in header: 7*4 + 4*8 + 4*4 + 28*8 = 300
        # ft3skipbytes(fid, 300)
        # Minus the added has_velocities attribute.
        ft3skipbytes(fid, 296)

        # ft3skipbytes(fid, 7 * 4 + 4 * 8 + 4 * 4 + 4 * 8)
        # has_velocities = ft3int(fid);
        # ft3skipbytes(fid, 4 + 23*8)

        # Skip phase fractions: nph * (nz+2) * (ny+2) * (nx+2)
        self.ncells = (self.nz + 2) * (self.ny + 2) * (self.nx + 2)
        ft3skipbytes(fid, 8 * self.ncells * self.nph)

        # Skip pressure: (nz+2) * (ny+2) * (nx+2)
        ft3skipbytes(fid, 8 * self.ncells)

        # Note: Skipping the velocity fields is easier than actually
        # reading them, since the staggered velocity requires 1 cell less
        # in the direction of the flow compared to the other directions,
        # but this missing cell is still present in the ft3 file (as a
        # dummy), to make sure that the field is still of size (nx+2)*(ny+2)*(nz+2).
        # So in case anyone wants to actually store the velocity fields,
        # refer to the original ft3 file format to properly accommodate
        # this!

        # Skip x-vel: (nz+2) * (ny+2) * (nx+2)
        # ft3skipbytes(fid,8 * self.ncells)
        xvel = np.fromfile(fid, dtype=float, count=self.ncells)

        # Skip y-vel: nph * (nz+2) * (ny+2) * (nx+2)
        # ft3skipbytes(fid, 8 * self.ncells)
        yvel = np.fromfile(fid, dtype=float, count=self.ncells)

        # Skip z-vel: nph * (nz+2) * (ny+2) * (nx+2)
        # ft3skipbytes(fid, 8 * self.ncells)
        zvel = np.fromfile(fid, dtype=float, count=self.ncells)

        self.velocity = np.reshape(
            np.column_stack((xvel, yvel, zvel)),
            (self.nx + 2, self.ny + 2, self.nz + 2, 3),
        )

        # We arrived at the bubble mesh definitions!
        for _ in range(self.neli):
            nmar = ft3int(fid)
            npos = ft3int(fid)
            # print('npos = ', npos)

            # Get the point positions from the file, reshape in a 3*npos array
            pointpos = np.reshape(
                np.fromfile(fid, dtype=float, count=npos * 3), (npos, 3)
            )

            if has_velocities:
                velocity = np.reshape(
                    np.fromfile(fid, dtype=float, count=npos * 3), (npos, 3)
                )
            else:
                velocity = []
                for pos in pointpos:
                    velocity.append(self.interpolateVelocity(pos))

            # The connectivity and point numbers are stored
            # alternatingly for each marker, e.g. for marker M
            # (connected_marker[M][0], point[M][0], connected_marker
            # [M][1], point[M][1], connected_marker[M][2], point[M]
            # [2]) --- The following first organizes these into 2
            # columns (1 for connected markers, the next for the
            # points), slices them into separate arrays, which are then
            # reshaped to hold 3 markers/points respectively for each
            # marker.
            connmrk = np.reshape(
                np.fromfile(fid, dtype=np.int32, count=nmar * 3 * 2),
                (nmar * 3, 2),
            )

            # Stores the neighboring cell indices.
            conn = np.reshape(connmrk[:, 0], (nmar, 3))

            # Stores the point indices that make up a cell.
            points = np.reshape(connmrk[:, 1], (nmar, 3))

            # Add the bubble as raw data to the array
            self.Bubbles.append(Bubble(pointpos, conn, points, velocity))

    def interpolateVelocity(self, pos):
        # print("interpolating")
        x = pos[0]
        y = pos[1]
        z = pos[2]

        x0 = int(x / self.dx)
        y0 = int(y / self.dy)
        z0 = int(z / self.dz)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # xd = (x - x0 * self.dx) / self.dx
        # yd = (y - y0 * self.dy) / self.dy
        # zd = (z - z0 * self.dy) / self.dz
        xd = x / self.dx - x0
        yd = y / self.dy - y0
        zd = z / self.dz - z0

        c00 = (
            self.velocity[z0, y0, x0] * (1 - xd)
            + self.velocity[z0, y0, x1] * xd
        )
        c01 = (
            self.velocity[z1, y0, x0] * (1 - xd)
            + self.velocity[z1, y0, x1] * xd
        )
        c10 = (
            self.velocity[z0, y1, x0] * (1 - xd)
            + self.velocity[z0, y1, x1] * xd
        )
        c11 = (
            self.velocity[z1, y1, x0] * (1 - xd)
            + self.velocity[z1, y1, x1] * xd
        )

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        return c0 * (1 - zd) + c1 * zd

    def getBubble(self, bubbleNumber):
        # Check if bubble number is within range, return arrays to position, marker and connectivity
        pass

    def plotBubblePoints(self, bubbleNumber):
        import matplotlib.pyplot as plt

        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.Bubbles[bubbleNumber].positions[:, 0],
            self.Bubbles[bubbleNumber].positions[:, 1],
            self.Bubbles[bubbleNumber].positions[:, 2],
        )
        plt.show()
