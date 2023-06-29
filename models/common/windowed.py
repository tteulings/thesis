from collections import deque
from typing import Iterable, Iterator, Tuple, TypeVar

T = TypeVar("T")


def windowed(
    seq: Iterable[T], n: int, step: int = 1
) -> Iterator[Tuple[T, ...]]:
    """Return a sliding window of width *n* over the given iterable.

        >>> all_windows = windowed([1, 2, 3, 4, 5], 3)
        >>> list(all_windows)
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    When the window is larger than the iterable, the result is empty:

        >>> list(windowed([1, 2, 3], 4))
        []

    Each window will advance in increments of *step*:

        >>> list(windowed([1, 2, 3, 4, 5, 6], 3, step=2))
        [(1, 2, 3), (3, 4, 5)]

    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        yield tuple()
        return
    if step < 1:
        raise ValueError("step must be >= 1")

    window = deque(maxlen=n)
    i = n
    for _ in map(window.append, seq):
        i -= 1
        if not i:
            i = step
            yield tuple(window)
