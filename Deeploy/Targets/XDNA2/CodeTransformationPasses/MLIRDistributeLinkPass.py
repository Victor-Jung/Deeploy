# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Device-phase pass that lowers a memtile-engine ``Split`` to an
``aie.objectfifo.link`` distribute pattern.

For one Split node colored to ``XDNA2MemTileExecutionEngine(col=c)`` with
``N`` consumer cores in column ``c``, this pass emits:

1. One big ``shim_c{c} → memtile_c{c}`` ObjectFifo carrying the full
   logical input buffer.
2. ``N`` small ``memtile_c{c} → core_c{c}_r{r_i}`` ObjectFifos, one per
   consumer chunk.
3. One ``aie.objectfifo.link [@big] -> [@small_0..N-1] ([] [0, k, 2k, …])``
   that wires the distribute geometry. Offsets are in elements.

Records every small-FIFO name into ``mlirBlock.fifoRegistry`` keyed by
``(logical_parent_name, col, row)`` so the consuming compute node's
``MLIRObjectFifoPass`` can look it up and skip creating its own
shim-to-core FIFO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import aie.ir as ir
import numpy as np
from aie.dialects import aie as aie_d

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext


def _resolveCoreCoords(engineName: str) -> Tuple[int, int]:
    """Parse "AIE_c{col}r{row}" → (col, row). Mirrors the convention from
    XDNA2AIECoreEngine.__init__'s default name.
    """
    assert engineName.startswith("AIE_c"), f"Unexpected engine name '{engineName}'"
    rest = engineName[len("AIE_c"):]
    col_str, row_str = rest.split("r")
    return int(col_str), int(row_str)


class MLIRDistributeLinkPass(MLIRCodeTransformationPass):

    def __init__(self, fifoDepth: int = 2) -> None:
        self.fifoDepth = fifoDepth

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        opRepr = mlirBlock.operatorRepresentation
        bigInputName = opRepr['data_in']

        # The Split's outputs are 'data_out_0', 'data_out_1', ..., contiguous
        # from 0. Walk until we run out of keys.
        chunkNames: List[str] = []
        idx = 0
        while True:
            key = f"data_out_{idx}"
            if key not in opRepr:
                break
            chunkNames.append(opRepr[key])
            idx += 1
        assert chunkNames, "Distribute pass: Split node has zero outputs."

        # Look up chunk buffers to recover their target core engine + size.
        chunkBufs = [ctxt.lookup(n) for n in chunkNames]
        # Defensive sort by chunk offset so the link offsets line up with
        # the chunk-to-core assignment regardless of node-output order.
        chunkBufs.sort(key = lambda b: getattr(b, "_chunkOffset", 0))

        # Per-chunk element count (uniform across chunks for equal split).
        elemTy = ir.BF16Type.get()

        def _flat(buf):
            sh = buf.shape
            if isinstance(sh, int):
                return int(sh)
            return int(np.prod(sh))

        chunkElems = _flat(chunkBufs[0])
        for b in chunkBufs[1:]:
            assert _flat(b) == chunkElems, (
                "Distribute pass currently assumes equal-split chunks; "
                f"got {_flat(b)} vs {chunkElems}.")

        # The small memtile-to-core FIFO carries ONE kernel-tile per
        # acquire (matching the consuming core's `objectfifo.acquire`
        # size), not the whole chunk. The big shim-to-memtile FIFO carries
        # one tile per consumer per acquire (so the link distributes a
        # single big object into N small tiles every cycle, and the
        # whole chunk streams through over chunk_elems / tile_elems
        # cycles).
        chunkTileElems = getattr(mlirBlock, "chunkTileElems", {}) or {}
        # Use the first consumer chunk's tile size; equal-split assumption
        # implies all consumers share it.
        firstChunkName = chunkBufs[0].name
        assert firstChunkName in chunkTileElems, (
            f"Distribute pass: no per-tile size known for chunk '{firstChunkName}'. "
            f"The deployer's _buildChunkTileElems should have populated it from "
            f"the consuming core's tiling solution.")
        tileElems = int(chunkTileElems[firstChunkName])
        assert chunkElems % tileElems == 0, (
            f"Distribute pass: chunk '{firstChunkName}' has {chunkElems} elements but "
            f"the consuming core's per-acquire tile is {tileElems} — not a clean divisor.")
        bigElems = tileElems * len(chunkBufs)
        bigBuf = ctxt.lookup(bigInputName)
        assert _flat(bigBuf) == chunkElems * len(chunkBufs), (
            f"Distribute pass: big input '{bigInputName}' has {_flat(bigBuf)} "
            f"elements but the chunks sum to {chunkElems * len(chunkBufs)}.")

        # Memref types: flat 1-D (matches existing MLIRObjectFifoPass).
        # Both bigTy and smallTy are sized to ONE acquire-cycle's worth
        # of bytes; the runtime sequence's shim DMA then transfers the
        # full chunk_elems × N elements over (chunk_elems / tile_elems)
        # objects' worth of FIFO traffic.
        bigTy = ir.MemRefType.get((bigElems,), elemTy)
        smallTy = ir.MemRefType.get((tileElems,), elemTy)

        # Resolve tiles. The deployer pre-populates these on the block.
        memTile = mlirBlock.computeTile     # the mem tile (we overload computeTile for memtile-engine nodes)
        shimTile = mlirBlock.shimTile       # the shim in this column
        coreTileMap = getattr(mlirBlock, "coreTileMap", None)
        assert coreTileMap is not None, (
            "Distribute pass: MLIRExecutionBlock.coreTileMap not populated by deployer.")

        prefix = name.replace(".", "_").replace("/", "_")

        # Big shim → memtile FIFO.
        bigFifoName = f"{prefix}_big"
        aie_d.object_fifo(bigFifoName, shimTile, [memTile], self.fifoDepth, bigTy)
        mlirBlock.fifoMap['data_in'] = bigFifoName
        mlirBlock.fifoTypes['data_in'] = bigTy

        # N small memtile → core FIFOs, one per chunk in offset order.
        smallNames: List[str] = []
        offsets: List[int] = []
        col = mlirBlock.tileCol
        for i, buf in enumerate(chunkBufs):
            targetEngine = getattr(buf, "_targetCoreEngine", None)
            assert targetEngine is not None, (
                f"Distribute pass: chunk '{buf.name}' has no _targetCoreEngine.")
            tgtCol, tgtRow = _resolveCoreCoords(targetEngine)
            assert tgtCol == col, (
                f"Distribute pass: chunk '{buf.name}' targets engine '{targetEngine}' "
                f"in col {tgtCol}, but mem tile is in col {col}.")
            assert targetEngine in coreTileMap, (
                f"Distribute pass: target engine '{targetEngine}' missing from coreTileMap.")
            destTile = coreTileMap[targetEngine]
            smallName = f"{prefix}_to_c{tgtCol}r{tgtRow}"
            aie_d.object_fifo(smallName, memTile, [destTile], self.fifoDepth, smallTy)
            smallNames.append(smallName)
            # Link offsets are in elements WITHIN the big object (per-cycle),
            # so they're per-tile, not per-chunk.
            offsets.append(i * tileElems)

            # Register so the consuming Add's MLIRObjectFifoPass can find it.
            mlirBlock.fifoRegistry[(bigInputName, tgtCol, tgtRow)] = smallName

        # The link op: one big producer, N small consumers, distribute by
        # dst-offset list. srcOffsets stays empty.
        aie_d.object_fifo_link(
            fifoIns = [bigFifoName],
            fifoOuts = smallNames,
            srcOffsets = [],
            dstOffsets = offsets,
        )

        return ctxt, mlirBlock
