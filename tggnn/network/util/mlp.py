from typing import List

from itertools import chain

from torch.nn import Sequential, ReLU, Linear, LayerNorm

from .weave import weave


def make_mlp(
    sizes: List[int], activate_final: bool = False, normalize: bool = True
) -> Sequential:
    layers = (Linear(i, o) for i, o in zip(sizes, sizes[1:]))
    activations = (
        ReLU() for _ in range(len(sizes) - (1 if activate_final else 2))
    )

    return Sequential(
        *chain(
            weave(layers, activations),
            [LayerNorm(sizes[-1])] if normalize else [],
        )
    )
