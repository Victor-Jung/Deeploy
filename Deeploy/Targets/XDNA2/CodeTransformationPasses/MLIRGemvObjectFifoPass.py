# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Device-phase pass: ObjectFifos + external kernel for BF16 GEMV.

GEMV ``y = A @ x`` with A=[M,N], x=[N,1], y=[M,1] on a single AIE core, using
the row-major ``mv.o`` microkernel (IRON generic ABI). N is the reduction
dimension K; each kernel call reduces over the **full** K, so K is not tiled —
only the M (output-row) dimension is tiled into ``DIM_M``-row bands.

Three ObjectFifos are created (shim <-> core, depth 2):

* ``inA``  : shim -> core, one ``(DIM_M, N)`` row-major tile per object. No
  transpose — the generic kernel reads A row-major directly.
* ``inB``  : shim -> core, the full ``N``-element vector, produced once and
  held on the core across all M tiles.
* ``outC`` : core -> shim, one ``DIM_M``-element output tile per object.

One external kernel is declared: ``matvec_vectorized_bf16_bf16`` with the IRON
ABI ``(i32 m, i32 row_offset, A_tile, x_vec, C_tile)``. No zero kernel is
needed because the generic kernel overwrites C.

Tile counts are stashed on the block for the compute / runtime passes:
``gemvNumMTiles = M / DIM_M`` (output tiles). There is no K-tile count — the
kernel consumes the whole N-vector per output tile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import aie.ir as ir
from aie.dialects import aie as aie_d

from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext

# Output-row tile granularity (rows computed per kernel call). Must divide M.
DIM_M = 32


class MLIRGemvObjectFifoPass(MLIRCodeTransformationPass):
    """Create GEMV ObjectFifos and declare the row-major matvec kernel."""

    def __init__(self, fifoDepth: int = 2) -> None:
        self.fifoDepth = fifoDepth

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        template = mlirBlock.template
        opRepr = mlirBlock.operatorRepresentation

        M = int(opRepr['M'])
        N = int(opRepr['N'])  # reduction dim K (consumed whole per kernel call)
        assert M % DIM_M == 0, f"[XDNA2 GEMV] M={M} not a multiple of DIM_M={DIM_M}"

        numMTiles = M // DIM_M

        bf16 = ir.BF16Type.get()
        i32 = ir.IntegerType.get_signless(32)
        aTy = ir.MemRefType.get((DIM_M, N), bf16)  # one A row band (2-D, row-major)
        bTy = ir.MemRefType.get((N,), bf16)         # full x vector
        cTy = ir.MemRefType.get((DIM_M,), bf16)     # one y tile

        computeTile = mlirBlock.computeTile
        shimTile = mlirBlock.shimTile
        prefix = name.replace(".", "_").replace("/", "_")

        aKey, bKey = template.INPUT_KEYS  # ['A', 'B']
        (cKey,) = template.OUTPUT_KEYS    # ['data_out']

        aFifo = f"{prefix}_inA"
        bFifo = f"{prefix}_inB"
        cFifo = f"{prefix}_outC"

        # A: shim -> core, plain row-major (no dimensionsFromStreamPerConsumer).
        aie_d.object_fifo(aFifo, shimTile, [computeTile], self.fifoDepth, aTy)
        aie_d.object_fifo(bFifo, shimTile, [computeTile], self.fifoDepth, bTy)
        aie_d.object_fifo(cFifo, computeTile, [shimTile], self.fifoDepth, cTy)

        mlirBlock.fifoMap[aKey] = aFifo
        mlirBlock.fifoMap[bKey] = bFifo
        mlirBlock.fifoMap[cKey] = cFifo
        mlirBlock.fifoTypes[aKey] = aTy
        mlirBlock.fifoTypes[bKey] = bTy
        mlirBlock.fifoTypes[cKey] = cTy

        mlirBlock.gemvNumMTiles = numMTiles
        mlirBlock.kernelFuncName = template.KERNEL_FN
        mlirBlock.kernelObjFile = template.KERNEL_OBJ

        # Declare matvec(i32 m, i32 row_offset, A_tile, x_vec, C_tile), once per device.
        declared = getattr(mlirBlock, "declaredKernels", None)
        if declared is None or template.KERNEL_FN not in declared:
            aie_d.external_func(template.KERNEL_FN, [i32, i32, aTy, bTy, cTy], link_with = template.KERNEL_OBJ)
            if declared is not None:
                declared.add(template.KERNEL_FN)

        log.info(f"[XDNA2 GEMV] ObjectFifo: M={M}, N(K)={N}, numMTiles={numMTiles}")
        return ctxt, mlirBlock
