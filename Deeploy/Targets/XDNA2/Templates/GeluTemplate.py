# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for BF16 GELU — pure compute primitive.

The kernel implements the tanh-based approximation
``0.5·x·(1 + tanh(sqrt(2/π)·(x + 0.044715·x³)))`` and uses the
``aie::tanh<bfloat16>`` intrinsic (no separate LUT needed; the
intrinsic is provided by ``aie_api`` and lives in the linked kernel
object).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aie.ir as ir
from aie.dialects import arith as arith_d
from aie.dialects import func as func_d

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2GeluTemplate(MLIRNodeTemplate):
    """Pure compute-primitive for BF16 GELU on XDNA2."""

    KERNEL_FN = "gelu_bf16"
    KERNEL_OBJ = "gelu.o"
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


referenceTemplate = XDNA2GeluTemplate()
