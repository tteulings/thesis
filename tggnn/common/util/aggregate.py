from enum import Enum


class Aggregate(str, Enum):
    SUM = "sum"
    MUL = "mul"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
