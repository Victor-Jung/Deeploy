# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for BF16 Tanh — pure compute primitive.

The kernel calls ``aie::tanh<bfloat16>``. As with GELU, the AIE BF16
tanh polynomial diverges from IEEE tanh in the saturation tail
(|x| > ~2); the test data generator clips inputs to keep comparisons
within tolerance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aie.ir as ir
from aie.dialects import arith as arith_d
from aie.dialects import func as func_d

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2TanhTemplate(MLIRNodeTemplate):
    """Pure compute-primitive for BF16 Tanh on XDNA2."""

    KERNEL_FN = "tanh_bf16"
    KERNEL_OBJ = "tanh.o"
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


referenceTemplate = XDNA2TanhTemplate()
