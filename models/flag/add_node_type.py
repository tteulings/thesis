import functools
import json
import sqlite3
from os import path

import tensorflow as tf
import torch

from tggnn.data.util import face_to_edge

from .data.util import NodeType, parse_tf_record
from .data.processing import tensor_to_binary

# root = "/home/TUE/s131727/git/meshgraphnets/data/flag_simple"
root = "/home/daan/git/flag-data/train"
record_name = "train.tfrecord"

with open(path.join(root, "meta.json")) as meta_file:
    meta = json.load(meta_file)

db = sqlite3.connect(path.join(root, "flag_static.db"))

db.execute('ALTER TABLE mesh ADD node_type BLOB NOT NULL DEFAULT ""')

node_type = torch.zeros((1579, 1), dtype=torch.long)
node_type[0] = NodeType.HANDLE
node_type[3] = NodeType.HANDLE

ds = tf.data.TFRecordDataset(path.join(root, record_name))
ds = ds.map(functools.partial(parse_tf_record, meta=meta), num_parallel_calls=1)

record = next(iter(ds))

cells = torch.tensor(record["cells"][0].numpy().transpose(), dtype=torch.long)

mesh_index = face_to_edge(cells)

cursor = db.cursor()

cursor.execute("BEGIN")
cursor.execute(
    "UPDATE mesh SET 'index' = ?, node_type = ?",
    [tensor_to_binary(mesh_index), tensor_to_binary(node_type)],
)
cursor.execute("COMMIT")
