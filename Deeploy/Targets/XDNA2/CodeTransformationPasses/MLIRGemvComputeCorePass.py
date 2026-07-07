# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Device-phase pass: emit the GEMV ``@aie_d.core`` with the M-tile loop.

Using the row-major generic kernel, each call computes ``DIM_M`` output rows by
reducing over the full K in one shot, so there is no K-reduction loop and no
zero-init. The core:

1. acquires the x vector once (Consume) and holds it across all M tiles,
2. loops over the ``numMTiles`` output tiles, each iteration:
   * acquires one C output tile (Produce),
   * acquires one ``(DIM_M, N)`` A tile (Consume),
   * calls ``matvec_vectorized_bf16_bf16(DIM_M, 0, A, x, C)`` (writes C),
   * releases the A and C tiles,
3. releases the x vector.

Mirrors the structure of IRON's gemv ``core_body`` with
``m_input == m_output == DIM_M`` (so ``row_offset`` is always 0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from aie.dialects import aie as aie_d
from aie.dialects import scf as scf_d

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRGemvObjectFifoPass import DIM_M

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext


class MLIRGemvComputeCorePass(MLIRCodeTransformationPass):
    """Emit ``@aie_d.core`` with the M-tile matvec loop (row-major kernel)."""

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        template = mlirBlock.template
        aKey, bKey = template.INPUT_KEYS
        (cKey,) = template.OUTPUT_KEYS

        computeTile = mlirBlock.computeTile
        numMTiles = mlirBlock.gemvNumMTiles
        opRepr = mlirBlock.operatorRepresentation

        aFifo = mlirBlock.fifoMap[aKey]
        bFifo = mlirBlock.fifoMap[bKey]
        cFifo = mlirBlock.fifoMap[cKey]
        aTy = mlirBlock.fifoTypes[aKey]
        bTy = mlirBlock.fifoTypes[bKey]
        cTy = mlirBlock.fifoTypes[cKey]

        aSubTy = aie_d.ObjectFifoSubviewType.get(aTy)
        bSubTy = aie_d.ObjectFifoSubviewType.get(bTy)
        cSubTy = aie_d.ObjectFifoSubviewType.get(cTy)

        Consume = aie_d.ObjectFifoPort.Consume
        Produce = aie_d.ObjectFifoPort.Produce

        @aie_d.core(computeTile)
        def _core():
            for _ in scf_d.for_(0, 0x7FFFFFFFFFFFFFFF, 1):
                # Acquire the x vector once; reuse it across every output tile.
                bAcq = aie_d.objectfifo_acquire(bSubTy, Consume, bFifo, 1)
                bVal = aie_d.objectfifo_subview_access(bTy, bAcq, 0)

                for _ in scf_d.for_(0, numMTiles, 1):
                    cAcq = aie_d.objectfifo_acquire(cSubTy, Produce, cFifo, 1)
                    cVal = aie_d.objectfifo_subview_access(cTy, cAcq, 0)
                    aAcq = aie_d.objectfifo_acquire(aSubTy, Consume, aFifo, 1)
                    aVal = aie_d.objectfifo_subview_access(aTy, aAcq, 0)

                    # One call computes DIM_M output rows over the full K; each
                    # C tile is exactly one m-band, so row_offset == 0.
                    modifiedOpRepr = {**opRepr, aKey: aVal, bKey: bVal, cKey: cVal, 'm': DIM_M, 'row_offset': 0}
                    template.emit(modifiedOpRepr)

                    aie_d.objectfifo_release(Consume, aFifo, 1)
                    aie_d.objectfifo_release(Produce, cFifo, 1)
                    scf_d.yield_([])

                aie_d.objectfifo_release(Consume, bFifo, 1)
                scf_d.yield_([])

        return ctxt, mlirBlock
