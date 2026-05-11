# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for BF16 elementwise Mul — pure compute primitive.

Mirrors :class:`XDNA2AddTemplate` byte-for-byte; only the kernel symbol
and object differ. Both ops have the same arity and tile-size signature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aie.ir as ir
from aie.dialects import arith as arith_d
from aie.dialects import func as func_d

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2MulTemplate(MLIRNodeTemplate):
    """Pure compute-primitive for BF16 elementwise Mul on XDNA2."""

    KERNEL_FN = "eltwise_mul_bf16_vector"
    KERNEL_OBJ = "mul.o"
    INPUT_KEYS = ['data_in_1', 'data_in_2']
    OUTPUT_KEYS = ['data_out']

    def __init__(self):
        super().__init__()

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        i32 = ir.IntegerType.get_signless(32)
        sizeVal = arith_d.constant(i32, int(operatorRepresentation['size']))
        func_d.call([], self.KERNEL_FN, [
            operatorRepresentation['data_in_1'],
            operatorRepresentation['data_in_2'],
            operatorRepresentation['data_out'],
            sizeVal,
        ])


referenceTemplate = XDNA2MulTemplate()
