from typing import List, Optional

from os import path

import torch

from tggnn.data.typed_graph import TypedGraphDataset, TypedGraphTransform

from .flag import Flag


class FlagSequence(TypedGraphDataset[Flag]):
    def __init__(
        self,
        root: str,
        sequence_idx: int,
        transforms: Optional[List[TypedGraphTransform[Flag]]] = None,
    ):
        super().__init__(root, transforms)

        self.idx = sequence_idx

        self.raw_dir = path.join(self.root, "raw")
        self.processed_dir = path.join(self.root, "processed")

        self.data: List[Flag] = torch.load(
            path.join(self.processed_dir, f"seq_{self.idx}.pt")
        )

    def __len__(self) -> int:
        return len(self.data)

    def __get__(self, idx: int) -> Flag:
        return self.data[idx]
