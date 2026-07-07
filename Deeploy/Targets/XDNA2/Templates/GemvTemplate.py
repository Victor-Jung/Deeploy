# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for BF16 GEMV (matrix-vector) — pure compute primitive.

``emit()`` emits a **single** ``func.call`` to ``matvec_vectorized_bf16_bf16``
(the IRON row-major generic kernel ABI): it computes ``m`` output rows of
``C = A @ x`` reducing over the full K dimension in one call, writing to
``C + row_offset``.

The surrounding structure — acquiring the x vector once and looping over the M
output tiles, acquiring one A tile and one C tile per iteration — is emitted by
:class:`MLIRGemvComputeCorePass`, so this template stays a pure compute
primitive (consistent with the other XDNA2 templates).

Unlike the older transposed kernel, the generic kernel:

* reads A in plain **row-major** order (no ObjectFifo transpose), and
* **overwrites** C (no cross-K accumulation), so no zero-init call is needed.

The reduction dim K (= ``DIM_K``) and the SIMD chunk size (``VEC_SIZE``) are
compile-time constants baked into ``mv.o``; ``m`` and ``row_offset`` are i32
runtime arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aie.ir as ir
from aie.dialects import arith as arith_d
from aie.dialects import func as func_d

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2GemvTemplate(MLIRNodeTemplate):
    """Pure compute-primitive for BF16 GEMV on XDNA2.

    ``emit()`` is called by :class:`MLIRGemvComputeCorePass` inside the M-tile
    loop, with ``operatorRepresentation`` entries replaced by live MLIR values:

    * ``A`` — acquired (m x K) row-major matrix tile (2-D memref).
    * ``B`` — acquired K-element vector tile (1-D memref), held across M tiles.
    * ``data_out`` — the acquired (m,) output tile (1-D memref).
    * ``m`` — number of output rows this call computes (python int).
    * ``row_offset`` — element offset into ``data_out`` to write from (python int).
    """

    KERNEL_FN = "matvec_vectorized_bf16_bf16"
    KERNEL_OBJ = "mv.o"
    INPUT_KEYS = ['A', 'B']
    OUTPUT_KEYS = ['data_out']

    def __init__(self):
        super().__init__()

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        """Emit one matvec call: C[row_offset:row_offset+m] = A_tile @ x (bf16)."""
        i32 = ir.IntegerType.get_signless(32)
        mVal = arith_d.constant(i32, int(operatorRepresentation['m']))
        offVal = arith_d.constant(i32, int(operatorRepresentation['row_offset']))
        func_d.call([], self.KERNEL_FN, [
            mVal,
            offVal,
            operatorRepresentation['A'],
            operatorRepresentation['B'],
            operatorRepresentation['data_out'],
        ])


referenceTemplate = XDNA2GemvTemplate()
