from dataclasses import dataclass
from typing import Optional

from ...modules.base_types import LabelNormalizer


@dataclass
class LabelFunctionSet:
    normalizer: Optional[LabelNormalizer]
