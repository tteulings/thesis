class SimulationConfig:
    def __init__(
        self,
        dx: float,
        dy: float,
        dz: float,
        dt: float,
        fak_min: float,
        fak_max: float,
    ):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.fak_min = fak_min
        self.fak_max = fak_max
