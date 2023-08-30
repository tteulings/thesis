from typing import Generic, TypeVar

import torch
from torch import Tensor

from ...data.typed_graph import TypedGraph, TypedGraphLayout
from ..container import ModuleDict, ModuleList
from ..graph import GraphBlock, GraphBlockImpl, GraphModule
from ..modules.base_types import DecoderImpl, EncoderImpl, NormalizerImpl
from ..util import AttributeModuleDict, SetModuleDict
from .config import EPDConfig
from .epoch import EpochIterator


TG_Data = TypeVar("TG_Data", bound="TypedGraph")


class EncodeProcessDecode(GraphModule[TG_Data], Generic[TG_Data]):
    epoch: Tensor

    _node_normalizers: ModuleDict[ModuleDict[NormalizerImpl]]
    _edge_normalizers: ModuleDict[NormalizerImpl]
    _label_normalizers: ModuleDict[NormalizerImpl]

    _node_encoders: ModuleDict[ModuleDict[EncoderImpl]]
    _edge_encoders: ModuleDict[EncoderImpl]

    _node_decoders: ModuleDict[ModuleDict[DecoderImpl]]
    _edge_decoders: ModuleDict[DecoderImpl]

    _blocks: ModuleList[GraphBlockImpl[TG_Data]]

    def __init__(
        self,
        config: EPDConfig,
        data_layout: TypedGraphLayout,
    ):
        super().__init__()

        (
            node_normalizer_inits,
            edge_normalizer_init,
            label_normalizer_init,
            node_encoder_inits,
            edge_encoder_init,
            node_decoder_inits,
            edge_decoder_init,
            block_init,
        ) = config.split_config(data_layout)

        # Initialize epoch counter
        self.register_buffer("epoch", torch.tensor([0], dtype=torch.int))

        # Prepare normalizers
        self._node_normalizers, data_layout = AttributeModuleDict(
            node_normalizer_inits
        ).init(data_layout)

        self._edge_normalizers, data_layout = SetModuleDict(
            edge_normalizer_init
        ).init(data_layout)

        self._label_normalizers, data_layout = SetModuleDict(
            label_normalizer_init
        ).init(data_layout)

        # Prepare input (encode) MLPs
        self._node_encoders, data_layout = AttributeModuleDict(
            node_encoder_inits
        ).init(data_layout)

        self._edge_encoders, data_layout = SetModuleDict(
            edge_encoder_init
        ).init(data_layout)

        # Prepare process blocks
        self._blocks = ModuleList()

        for block_config in block_init:
            block, data_layout = GraphBlock[TG_Data]()(
                block_config, data_layout
            )
            self._blocks.append(block)

        # Prepare output (decode) MLPs
        self._node_decoders, data_layout = AttributeModuleDict(
            node_decoder_inits
        ).init(data_layout)

        self._edge_decoders, data_layout = SetModuleDict(
            edge_decoder_init
        ).init(data_layout)

    def take_epoch(self, epochs: int) -> EpochIterator[TG_Data]:
        return EpochIterator(self, epochs)

    def gather_statistics(self, data: TypedGraph) -> None:
        for name, attrs in self._node_normalizers.items():
            for attr, normalizer in attrs.items():
                normalizer.accumulate(data.node_sets[name][attr].attr)

        for name, normalizer in self._edge_normalizers.items():
            edge_attr = data.edge_sets[name].attr

            # FIXME: This exception should occur in the constructor.
            if edge_attr is None:
                raise Exception(
                    f'Trying to call normalizer on edge set "{name}" that does'
                    " not have edge attributes."
                )

            normalizer.accumulate(edge_attr)

        for name, normalizer in self._label_normalizers.items():
            normalizer.accumulate(data.labels[name])

    def forward(self, data: TG_Data) -> TG_Data:
        # Normalize
        for name, attrs in self._node_normalizers.items():
            for attr, normalizer in attrs.items():
                print(name, attr)
                data.node_sets[name][attr].attr = normalizer(
                    data.node_sets[name][attr].attr
                )

        for name, normalizer in self._edge_normalizers.items():
            edge_attr = data.edge_sets[name].attr

            # FIXME: This exception should occur in the constructor.
            if edge_attr is None:
                raise Exception(
                    f'Trying to call normalizer on edge set "{name}" that does'
                    " not have edge attributes."
                )

            data.edge_sets[name].attr = normalizer(edge_attr)

        for name, normalizer in self._label_normalizers.items():
            data.labels[name] = normalizer(data.labels[name])

        old_velocity = data.node_sets['centroid']['velocity'].attr
        # Encode
        for name, attrs in self._node_encoders.items():
            
            for attr, encoder in attrs.items():
                # print(name, attr)
                data.node_sets[name][attr].attr = encoder(
                    data.node_sets[name][attr].attr
                )
        old_velocity2 = data.node_sets['centroid']['velocity'].attr

        for name, encoder in self._edge_encoders.items():
            data.edge_sets[name].attr = encoder(data.edge_sets[name].attr)

        # print('centroid_decoder', data.node_sets['centroid']['memory'].attr)

        # Process
        for block in self._blocks:
            data = block.forward(data)
        # print("OLD", old_velocity)
        # Decode
        for name, decoder in self._edge_decoders.items():
            edge_attr = data.edge_sets[name].attr

            if edge_attr is None:
                raise Exception(
                    f'Trying to call decoder on edge set "{name}" that does'
                    " not have edge attributes."
                )

            data.edge_sets[name].attr = decoder.forward(edge_attr)

        for name, attrs in self._node_decoders.items():
            for attr, decoder in attrs.items():
                if name == 'centroid':
                    # print('centroid_decoder', data.node_sets[name][attr].attr)
                    # print(data.node_sets[name]['memory'].attr)

                    # print('TESTER', old_velocity2, data.node_sets[name][attr].attr)
                 # # Angle 45
                    center_output = decoder.forward(
                        torch.cat(( data.node_sets[name]['memory'].attr, old_velocity), 1) 
                    )

                    # # Angle
                    # center_output = decoder.forward(
                    #     torch.cat((data.node_sets[name][attr].attr,  data.node_sets[name]['memory'].attr), 1) 
                    # )
                    # center_output = decoder.forward(
                    #     torch.cat(( data.node_sets[name]['memory'].attr, old_velocity), 1) 
                    # )
                    print('output', center_output)
                else:
                    data.node_sets[name][attr].attr = decoder.forward(
                        data.node_sets[name][attr].attr
                    )
        return data,center_output
