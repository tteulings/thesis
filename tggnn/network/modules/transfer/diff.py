# FIXME: Should be reworked to work with node attributes.

# from typing import List, Optional, Tuple

# import torch
# from torch import Tensor
# from torch.nn import Sequential

# from ....data.typed_graph import TypedGraphLayout
# from ...util import make_mlp
# from .base import TransferModule, TransferModuleImpl


# class DiffTransferImpl(TransferModuleImpl):
#     mlp: Sequential

#     def __init__(
#         self,
#         layout: List[int],
#         activate_final: bool = False,
#         normalize: bool = True,
#     ) -> None:
#         super().__init__()

#         self.mlp = make_mlp(layout, activate_final, normalize)

#     def forward(
#         self,
#         source: Tensor,
#         target: Tensor,
#         edge_attr: Optional[Tensor],
#     ) -> Tensor:
#         return self.mlp(
#             torch.cat(
#                 [target - source]
#                 + ([edge_attr] if edge_attr is not None else []),
#                 1,
#             )
#         )


# class DiffTransfer(TransferModule):
#     _hiddens: List[int]
#     _activate_final: bool
#     _normalize: bool

#     def __init__(
#         self,
#         hiddens: List[int],
#         activate_final: bool = False,
#         normalize: bool = True,
#     ) -> None:
#         self._hiddens = hiddens
#         self._activate_final = activate_final
#         self._normalize = normalize

#     def __call__(
#         self, edge_key: str, layout: TypedGraphLayout
#     ) -> Tuple[TransferModuleImpl, TypedGraphLayout]:
#         edge_set = layout.edge_sets[edge_key]

#         source_attrs = layout.node_sets[edge_set.source].attrs
#         target_attrs = layout.node_sets[edge_set.target].attrs

#         if source_attrs != target_attrs:
#             raise Exception(
#                 f'Cannot instantiate DiffTransfer for edge set "{edge_key}".'
#                 f" The number of source node attributes ({source_attrs}) is not"
#                 " equal to the number of target node attributes"
#                 f" ({target_attrs})."
#             )

#         input_size = edge_set.attrs + target_attrs

#         layout.edge_sets[edge_key].attrs = self._hiddens[-1]

#         return (
#             DiffTransferImpl(
#                 [input_size] + self._hiddens,
#                 self._activate_final,
#                 self._normalize,
#             ),
#             layout,
#         )
