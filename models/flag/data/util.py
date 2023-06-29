from collections import deque
import enum
from typing import Sequence, Tuple, TypeVar, Iterator

import tensorflow as tf


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def parse_tf_record(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {
        k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]
    }
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(
            features[key].values, getattr(tf, field["dtype"])
        )
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(
                features["length_" + key].values, tf.int32
            )
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out
