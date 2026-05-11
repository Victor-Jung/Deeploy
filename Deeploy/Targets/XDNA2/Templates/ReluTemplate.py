# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for BF16 ReLU — pure compute primitive.

Mirrors :class:`XDNA2SiLUTemplate`; only the kernel symbol/object differ.
ReLU's kernel uses a 32-element vectorisation factor, so the tile size
emitted by the tiler must be a multiple of 32 (true for the default
4096-element tiling on the 1024×1024 reference workload).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aie.ir as ir
from aie.dialects import arith as arith_d
from aie.dialects import func as func_d

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2ReluTemplate(MLIRNodeTemplate):
    """Pure compute-primitive for BF16 ReLU on XDNA2."""

    KERNEL_FN = "relu_bf16"
    KERNEL_OBJ = "relu.o"
    INPUT_KEYS = ['data_in']
    OUTPUT_KEYS = ['data_out']

    def __init__(self):
        super().__init__()

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        i32 = ir.IntegerType.get_signless(32)
        sizeVal = arith_d.constant(i32, int(operatorRepresentation['size']))
        func_d.call([], self.KERNEL_FN, [
            operatorRepresentation['data_in'],
            operatorRepresentation['data_out'],
            sizeVal,
        ])


referenceTemplate = XDNA2ReluTemplate()
