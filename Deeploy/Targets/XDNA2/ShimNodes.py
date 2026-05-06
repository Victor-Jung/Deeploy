# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""ONNX-level ops representing data movement on AIE shim tiles.

The XDNA2 spatial-mapping abstraction encodes every memory transition as a
node in the graph. Inputs live in L3 (DRAM) — to feed an AIE compute core
they must be DMA'd through a shim tile into a compute-tile-local L1
ObjectFifo. Symmetrically, outputs DMA back from L1 through the shim into
L3.

We model these transitions with two custom ops:

* ``ShimRead(L3_src) -> L1_chunk``: configure a shim DMA that reads
  ``length`` elements starting at ``offset`` of ``L3_src`` and pushes them
  into the ObjectFifo whose consumer is the downstream compute core.
* ``ShimWrite(L1_chunk) -> L3_dst``: configure a shim DMA that drains the
  ObjectFifo whose producer is the upstream compute core and writes
  ``length`` elements at ``offset`` into ``L3_dst``.

These nodes are pure metadata anchors: they're colored to a
:class:`XDNA2ShimEngine` (so memory-level resolution naturally returns L3),
they get :class:`UntiledTileConstraint` so the tiler keeps them at full
shape, and their MLIR template emits nothing. The surrounding compute
node's :class:`MLIRRuntimeSequencePass` walks the graph one hop to read
``offset`` / ``length`` / target-tensor from these nodes and emits the
actual DMA descriptors as part of its per-compute-block sequence.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer, PointerClass
from Deeploy.CommonExtensions.DataTypes import bfloat16_t
from Deeploy.CommonExtensions.TypeCheckers.SignPropTypeChecker import SignPropTypeChecker
from Deeploy.DeeployTypes import NetworkContext, NodeBinding, NodeMapper, NodeParser, ONNXLayer, OperatorRepresentation, \
    VariableBuffer
from Deeploy.MLIRDataTypes import MLIRCodeTransformation
from Deeploy.Targets.Generic.Layers import ConcatLayer
from Deeploy.Targets.Generic.Parsers import ConcatParser
from Deeploy.Targets.Generic.TileConstraints.UntiledTileConstraint import UntiledTileConstraint
from Deeploy.Targets.XDNA2.Templates import ShimNodeTemplate
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings


class XDNA2ShimChecker(SignPropTypeChecker):
    """BF16 in / BF16 out, single-input single-output."""

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [1]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [True]


class _ShimNodeParser(NodeParser):
    """Common parser for ``ShimRead`` / ``ShimWrite``.

    Both ops carry the same attribute set:

    * ``axis`` — split axis (default 0). Reserved for future N-D layouts.
    * ``offset`` — element offset within the L3 tensor.
    * ``length`` — number of elements transferred.
    """

    def parseNode(self, node: gs.Node) -> bool:
        return all([
            "offset" in node.attrs,
            "length" in node.attrs,
            len(node.inputs) == 1,
            len(node.outputs) == 1,
        ])

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation["data_in"] = data_in.name
        self.operatorRepresentation["data_out"] = data_out.name
        self.operatorRepresentation["offset"] = int(node.attrs["offset"])
        self.operatorRepresentation["length"] = int(node.attrs["length"])
        self.operatorRepresentation["axis"] = int(node.attrs.get("axis", 0))
        return ctxt, True


class ShimReadParser(_ShimNodeParser):
    pass


class ShimWriteParser(_ShimNodeParser):
    pass


class ShimReadLayer(ONNXLayer):
    pass


class ShimWriteLayer(ONNXLayer):
    pass


# Shim nodes emit no MLIR themselves. The surrounding compute node owns DMA
# emission and reads (offset, length, target_tensor) from these nodes via
# graph traversal. Empty transformer = no device or runtime-sequence passes.
_shimTransformer = MLIRCodeTransformation(devicePasses = [], runtimeSequencePasses = [])

XDNA2ShimReadBindings = [
    NodeBinding(
        XDNA2ShimChecker([PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        ShimNodeTemplate.referenceTemplate,
        _shimTransformer,
    )
]

XDNA2ShimWriteBindings = [
    NodeBinding(
        XDNA2ShimChecker([PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        ShimNodeTemplate.referenceTemplate,
        _shimTransformer,
    )
]

XDNA2ShimReadMapper = NodeMapper(
    ShimReadParser(),
    TilingReadyNodeBindings(nodeBindings = XDNA2ShimReadBindings, tileConstraint = UntiledTileConstraint()),
)
XDNA2ShimWriteMapper = NodeMapper(
    ShimWriteParser(),
    TilingReadyNodeBindings(nodeBindings = XDNA2ShimWriteBindings, tileConstraint = UntiledTileConstraint()),
)

# Concat marker added by the spatial split pass when num_chunks > 1 to keep
# the graph single-producer. It carries no DMA params — the deployer just
# walks one hop forward through it to find the graph output. We register one
# binding per supported chunk count: 2 through MAX_AIE_CORES inclusive,
# where MAX_AIE_CORES (32) is the AIE core count of the XDNA2 NPU.
MAX_AIE_CORES = 32


def _concatBindings(num_chunks: int):
    return [
        NodeBinding(
            XDNA2ShimChecker([PointerClass(bfloat16_t)] * num_chunks, [PointerClass(bfloat16_t)]),
            ShimNodeTemplate.referenceTemplate,
            _shimTransformer,
        )
    ]


XDNA2ConcatMappers = [
    NodeMapper(ConcatParser(),
               TilingReadyNodeBindings(nodeBindings = _concatBindings(n), tileConstraint = UntiledTileConstraint()))
    for n in range(2, MAX_AIE_CORES + 1)
]
