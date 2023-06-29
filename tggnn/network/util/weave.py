from itertools import cycle, islice
from typing import Generator, Iterable, Any


def weave(*iterables: Iterable[Any]) -> Generator[Any, None, None]:
    num_active = len(iterables)
    values = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for value in values:
                yield value()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            values = cycle(islice(values, num_active))
