from typing import Optional, Tuple

import os
from concurrent.futures import ThreadPoolExecutor, Future

import torch

from tggnn.data.typed_graph import (
    TypedGraphLayout,
    TypedGraphProcessor,
)

from .bin import BinFile
from .bubble import Bubble
from .config import SimulationConfig
from .processing import BubbleProcessConfig


def store_cycle(
    path: str,
    cycle: int,
    config: SimulationConfig,
    file_names: Tuple[str, str, str],
) -> Bubble:
    prev = BinFile(file_names[0])
    prev.load()

    data = BinFile(file_names[1])
    data.load()

    label = BinFile(file_names[2])
    label.load()

    bubble = Bubble(
        config, (cycle % 100) == 0, prev.volume(), prev, data, label
    )

    torch.save(bubble, path)

    return bubble


class BubbleProcessor(TypedGraphProcessor):
    def __init__(self, root: str, config: BubbleProcessConfig) -> None:
        super().__init__(root)

        self.config = config

        self.raw_dir = os.path.join(root, "raw")
        self.processed_dir = os.path.join(root, "processed")
        self.layout: Optional[TypedGraphLayout] = None

    def __process__(self) -> TypedGraphLayout:
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)

        # NOTE: Discard the first 0.2 seconds (Roghair et. al 2016, par 2.7,
        # fig 6b) Define preliminary sequential samplers
        prev_names = (
            f"{self.raw_dir}/F{i}_{'post' if i % 100 == 0 else 'pre'}.bin"
            for i in range(19990, 99990, 10)
        )
        data_names = (
            f"{self.raw_dir}/F{i}_pre.bin" for i in range(20000, 100000, 10)
        )
        label_names = (
            f"{self.raw_dir}/F{i}_pre.bin" for i in range(20010, 100010, 10)
        )

        with ThreadPoolExecutor(
            max_workers=self.config.num_workers
        ) as executor:
            first = True

            for i, file_names in enumerate(
                zip(prev_names, data_names, label_names)
            ):
                future = executor.submit(
                    store_cycle,
                    os.path.join(self.processed_dir, f"F{i}.pt"),
                    i,
                    self.config.sim_config,
                    file_names,
                )

                if first:
                    first = False

                    def set_layout(future: Future[Bubble]):
                        self.layout = future.result().layout()

                    future.add_done_callback(set_layout)

        if self.layout is None:
            raise Exception(
                "Could not obtain layout while processing Bubble data."
            )

        return self.layout
