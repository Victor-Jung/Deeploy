# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""No-op MLIR template for shim-colored ``ShimRead`` / ``ShimWrite`` nodes.

These nodes are metadata anchors: they carry the ``(offset, length, target)``
DMA parameters as ONNX attributes, but the actual ``aiex.dma_*`` emission
is owned by the surrounding compute node's runtime-sequence pass (which
walks one hop in the graph to find them). This template's ``emit`` is a
no-op; the deployer dispatches on engine kind and skips shim-colored nodes
entirely during MLIR generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2ShimNodeTemplate(MLIRNodeTemplate):

    KERNEL_FN = ""
    KERNEL_OBJ = ""
    INPUT_KEYS = ["data_in"]
    OUTPUT_KEYS = ["data_out"]

    def __init__(self):
        super().__init__()

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        return


referenceTemplate = XDNA2ShimNodeTemplate()
