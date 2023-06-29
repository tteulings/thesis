import collections.abc as container_base
from typing import (
    Generic,
    Mapping,
    Optional,
    ItemsView,
    Iterable,
    Iterator,
    OrderedDict,
    TypeVar,
    ValuesView,
)

from torch.nn import Module


T = TypeVar("T", bound="Module")


class ModuleDict(Generic[T], Module):
    r"""Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~torch.nn.ModuleDict` (the argument to
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules: Optional[Mapping[str, T]] = None) -> None:
        super().__init__()

        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> T:
        if key not in self._modules:
            raise KeyError(key)

        return self._modules[key]  # type: ignore

    def __setitem__(self, key: str, module: T) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        self._modules.clear()

    def pop(self, key: str) -> T:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (string): key to pop from the ModuleDict
        """
        value = self[key]
        del self[key]
        return value

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys."""
        return self._modules.keys()

    def items(self) -> ItemsView[str, T]:
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return self._modules.items()  # type: ignore

    def values(self) -> ValuesView[T]:
        r"""Return an iterable of the ModuleDict values."""
        return self._modules.values()  # type: ignore

    def update(self, modules: Mapping[str, T]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(modules, container_base.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an "
                "iterable of key/value pairs, but got "
                + type(modules).__name__
            )

        if isinstance(
            modules, (OrderedDict, ModuleDict, container_base.Mapping)
        ):
            for key, module in modules.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, module in enumerate(modules):
                if not isinstance(module, container_base.Collection):
                    raise TypeError(
                        "ModuleDict update sequence element #"
                        + str(j)
                        + " should be Iterable; is"
                        + type(module).__name__
                    )
                if not len(module) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element #"
                        + str(j)
                        + " has length "
                        + str(len(module))
                        + "; 2 is required"
                    )
                # modules can be Mapping (what it's typed at), or a list:
                # [(name1, module1), (name2, module2)] that's too cumbersome to
                # type correctly with overloads, so we add an ignore here
                self[module[0]] = module[1]  # type: ignore[assignment]
