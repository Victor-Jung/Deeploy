# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Device-phase pass that lowers a memtile-engine ``Concat`` to an
``aie.objectfifo.link`` join pattern.

Symmetric to :class:`MLIRDistributeLinkPass`: ``N`` core → memtile small
ObjectFifos, one big memtile → shim ObjectFifo, one
``aie.objectfifo.link [@small_0..N-1] -> [@big] ([0, k, 2k, …] [])`` join op.
Records small FIFO names into ``mlirBlock.fifoRegistry`` so the producing
compute node's ``MLIRObjectFifoPass`` can reuse them instead of emitting
its own core→shim FIFO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import aie.ir as ir
import numpy as np
from aie.dialects import aie as aie_d

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRDistributeLinkPass import _resolveCoreCoords

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext


class MLIRJoinLinkPass(MLIRCodeTransformationPass):

    def __init__(self, fifoDepth: int = 2) -> None:
        self.fifoDepth = fifoDepth

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        opRepr = mlirBlock.operatorRepresentation
        bigOutputName = opRepr['data_out']

        # ConcatParser populates 'data_in_1', 'data_in_2', ... contiguous from 1.
        chunkNames: List[str] = []
        idx = 1
        while True:
            key = f"data_in_{idx}"
            if key not in opRepr:
                break
            chunkNames.append(opRepr[key])
            idx += 1
        assert chunkNames, "Join pass: Concat node has zero inputs."

        chunkBufs = [ctxt.lookup(n) for n in chunkNames]
        chunkBufs.sort(key = lambda b: getattr(b, "_chunkOffset", 0))

        elemTy = ir.BF16Type.get()

        def _flat(buf):
            sh = buf.shape
            if isinstance(sh, int):
                return int(sh)
            return int(np.prod(sh))

        chunkElems = _flat(chunkBufs[0])
        for b in chunkBufs[1:]:
            assert _flat(b) == chunkElems, (
                "Join pass currently assumes equal-split chunks; "
                f"got {_flat(b)} vs {chunkElems}.")

        # Per-cycle tile size from the producing core's tiling solution.
        # See MLIRDistributeLinkPass for the rationale.
        chunkTileElems = getattr(mlirBlock, "chunkTileElems", {}) or {}
        firstChunkName = chunkBufs[0].name
        assert firstChunkName in chunkTileElems, (
            f"Join pass: no per-tile size known for chunk '{firstChunkName}'.")
        tileElems = int(chunkTileElems[firstChunkName])
        assert chunkElems % tileElems == 0, (
            f"Join pass: chunk '{firstChunkName}' has {chunkElems} elements but "
            f"the producing core's per-acquire tile is {tileElems} — not a clean divisor.")
        bigElems = tileElems * len(chunkBufs)
        bigBuf = ctxt.lookup(bigOutputName)
        assert _flat(bigBuf) == chunkElems * len(chunkBufs), (
            f"Join pass: big output '{bigOutputName}' has {_flat(bigBuf)} "
            f"elements but the chunks sum to {chunkElems * len(chunkBufs)}.")

        bigTy = ir.MemRefType.get((bigElems,), elemTy)
        smallTy = ir.MemRefType.get((tileElems,), elemTy)

        memTile = mlirBlock.computeTile
        shimTile = mlirBlock.shimTile
        coreTileMap = getattr(mlirBlock, "coreTileMap", None)
        assert coreTileMap is not None, (
            "Join pass: MLIRExecutionBlock.coreTileMap not populated by deployer.")

        prefix = name.replace(".", "_").replace("/", "_")
        col = mlirBlock.tileCol

        smallNames: List[str] = []
        offsets: List[int] = []
        for i, buf in enumerate(chunkBufs):
            producerOpRepr = ctxt.lookupProducerOpRepr(buf.name)
            srcEngine = producerOpRepr.get("engine")
            assert srcEngine is not None, (
                f"Join pass: chunk '{buf.name}' producer "
                f"'{producerOpRepr.get('nodeName')}' has no 'engine' in its "
                f"operatorRepresentation. EngineColoringPass should have set it.")
            srcCol, srcRow = _resolveCoreCoords(srcEngine)
            assert srcCol == col, (
                f"Join pass: chunk '{buf.name}' came from engine '{srcEngine}' "
                f"in col {srcCol}, but mem tile is in col {col}.")
            assert srcEngine in coreTileMap, (
                f"Join pass: source engine '{srcEngine}' missing from coreTileMap.")
            srcTile = coreTileMap[srcEngine]
            smallName = f"{prefix}_from_c{srcCol}r{srcRow}"
            aie_d.object_fifo(smallName, srcTile, [memTile], self.fifoDepth, smallTy)
            smallNames.append(smallName)
            # Per-cycle (per-tile) offsets within the big object.
            offsets.append(i * tileElems)

            # Register so the producing Add's MLIRObjectFifoPass can find it.
            mlirBlock.fifoRegistry[(bigOutputName, srcCol, srcRow)] = smallName

        bigFifoName = f"{prefix}_big"
        aie_d.object_fifo(bigFifoName, memTile, [shimTile], self.fifoDepth, bigTy)
        mlirBlock.fifoMap['data_out'] = bigFifoName
        mlirBlock.fifoTypes['data_out'] = bigTy

        # The link op: N small producers → 1 big consumer, join by
        # src-offset list. dstOffsets stays empty.
        aie_d.object_fifo_link(
            fifoIns = smallNames,
            fifoOuts = [bigFifoName],
            srcOffsets = offsets,
            dstOffsets = [],
        )

        return ctxt, mlirBlock
