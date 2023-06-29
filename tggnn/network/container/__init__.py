# NOTE: This module contains pytorch containers, but with a generic argument
# (T, bound by Module) that specifies the type of module stored by the
# container.  This is useful e.g. when explicitly storing GraphModules, that
# have a different `forward` function signature.

from .module_dict import ModuleDict
from .module_list import ModuleList
