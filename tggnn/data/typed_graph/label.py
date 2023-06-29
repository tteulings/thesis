from torch import Tensor


class LabelSummary:
    attrs: int

    def __init__(self, label: Tensor) -> None:
        self.attrs = label.size()[1]

    def __eq__(self, other: "LabelSummary") -> bool:
        return self.attrs == other.attrs

    def __repr__(self) -> str:
        return str(self.__dict__)
