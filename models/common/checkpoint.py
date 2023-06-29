from dataclasses import dataclass
from typing import Any, Dict, Optional, OrderedDict


@dataclass
class Checkpoint:
    model_state_dict: OrderedDict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
