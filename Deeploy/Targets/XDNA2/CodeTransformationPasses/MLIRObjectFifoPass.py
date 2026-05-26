# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Device-phase pass that creates ObjectFifos and declares external kernels.

Given an :class:`MLIRExecutionBlock` with ``computeTile``, ``shimTile``,
``operatorRepresentation``, and (optionally) ``patternMemoryConstraint``,
this pass:

1. Derives ``tileSize`` and ``numTiles`` (from tiling solver or fallback).
2. Creates one ``aie_d.object_fifo`` per input tensor (shim → compute)
   and one per output tensor (compute → shim), all with depth 2
   (double-buffering).
3. Declares the external kernel via ``aie_d.external_func``.
4. Stores FIFO names, types, and kernel metadata on the block for
   downstream passes and the compute template.

The pass is operator-agnostic — it only needs the tensor names and a
tile-size derivation function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import aie.ir as ir
import numpy as np
from aie.dialects import aie as aie_d

from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext

MAX_TILE_SIZE = 1024


def _deriveTileShape(numElements: int, patternMemoryConstraint) -> Tuple[int, ...]:
    """Extract the N-D tile shape from the tiling solution."""

    nodeConstraint = patternMemoryConstraint.nodeConstraints[0]
    outputConstraints = nodeConstraint.outputTensorMemoryConstraints
    if outputConstraints:
        firstOutputName = list(outputConstraints.keys())[0]
        tensorConstraint = outputConstraints[firstOutputName]
        for levelName, levelConstraint in tensorConstraint.memoryConstraints.items():
            if levelName.startswith("L1") and levelConstraint.shape is not None:
                return tuple(int(d) for d in levelConstraint.shape)

    # Debug aid before failing.
    debug_keys = list(tensorConstraint.memoryConstraints.keys()) if outputConstraints else []
    raise ValueError(f"_deriveTileShape: no L1* memory constraint found. Available levels for "
                     f"'{firstOutputName if outputConstraints else None}': {debug_keys}")


class MLIRObjectFifoPass(MLIRCodeTransformationPass):
    """Create ObjectFifos and declare the external kernel.

    All operator-specific metadata (tensor keys, kernel function name,
    kernel object file, argument types) is read from the
    :class:`MLIRNodeTemplate` stored on ``mlirBlock.template``.

    Parameters
    ----------
    fifoDepth : int
        ObjectFifo depth (default 2 for double-buffering).
    """

    def __init__(self, fifoDepth: int = 2) -> None:
        self.fifoDepth = fifoDepth

    @staticmethod
    def _registryHit(mlirBlock: MLIRExecutionBlock, ctxt: NetworkContext,
                     tensorName: str) -> Optional[Tuple[str, str]]:
        """Look up a memtile-core FIFO created upstream by a Split/Concat node.

        Returns ``(fifoName, registryKey)`` or ``None`` when no hit. The
        registry is keyed by ``(logical_parent, col, row)`` — that's the
        chunk's provenance plus the consumer / producer core's tile.
        """
        registry = getattr(mlirBlock, "fifoRegistry", None)
        if not registry:
            return None
        try:
            buf = ctxt.lookup(tensorName)
        except KeyError:
            return None
        parent = getattr(buf, "_logicalParent", None)
        col = getattr(mlirBlock, "tileCol", None)
        row = getattr(mlirBlock, "tileRow", None)
        if parent is None or col is None or row is None:
            return None
        regKey = (parent, col, row)
        if regKey in registry:
            return registry[regKey], regKey
        return None

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        template = mlirBlock.template
        inputTensorKeys = template.INPUT_KEYS
        outputTensorKeys = template.OUTPUT_KEYS

        opRepr = mlirBlock.operatorRepresentation
        numElements = int(opRepr['size'])

        # Read tile shape from the tiling solver.  The tile constraints
        # are the sole authority on valid shapes.
        tileShape = _deriveTileShape(numElements, mlirBlock.patternMemoryConstraint)
        tileSize = int(np.prod(tileShape))

        assert numElements % tileSize == 0, (f"[XDNA2] Tile size {tileSize} (shape {tileShape}) does not evenly "
                                             f"divide numElements {numElements}.  Fix the tile constraint.")

        numTiles = numElements // tileSize

        mlirBlock.tileShape = tileShape
        mlirBlock.tileSize = tileSize
        mlirBlock.numTiles = numTiles
        mlirBlock.numElements = numElements
        mlirBlock.kernelFuncName = template.KERNEL_FN
        mlirBlock.kernelObjFile = template.KERNEL_OBJ

        log.info(f"[XDNA2] ObjectFifo: tileShape={tileShape}, tileSize={tileSize}, "
                 f"numTiles={numTiles}, numElements={numElements}")

        # ObjectFifo memref is 1-D (flat) — the DMA and kernels work on flat buffers.
        tileTy = ir.MemRefType.get((tileSize,), ir.BF16Type.get())
        computeTile = mlirBlock.computeTile
        shimTile = mlirBlock.shimTile

        # Per-node FIFO name prefix so multiple compute cores can coexist in
        # the same @aie_d.device block without name collisions. ``name`` here
        # is the deployer-supplied node name.
        prefix = name.replace(".", "_").replace("/", "_")

        # Create input ObjectFifos (shim → compute), unless a Split node
        # upstream already created a memtile→core FIFO for this chunk.
        for idx, key in enumerate(inputTensorKeys):
            tensorName = opRepr.get(key)
            hit = self._registryHit(mlirBlock, ctxt, tensorName) if tensorName else None
            if hit is not None:
                mlirBlock.fifoMap[key] = hit[0]
                mlirBlock.fifoTypes[key] = tileTy
                continue
            fifoName = f"{prefix}_in{idx + 1}"
            aie_d.object_fifo(fifoName, shimTile, [computeTile], self.fifoDepth, tileTy)
            mlirBlock.fifoMap[key] = fifoName
            mlirBlock.fifoTypes[key] = tileTy

        # Create output ObjectFifos (compute → shim), unless a Concat
        # node downstream already created a core→memtile FIFO.
        for idx, key in enumerate(outputTensorKeys):
            tensorName = opRepr.get(key)
            hit = self._registryHit(mlirBlock, ctxt, tensorName) if tensorName else None
            if hit is not None:
                mlirBlock.fifoMap[key] = hit[0]
                mlirBlock.fifoTypes[key] = tileTy
                continue
            fifoName = f"{prefix}_out{idx}"
            aie_d.object_fifo(fifoName, computeTile, [shimTile], self.fifoDepth, tileTy)
            mlirBlock.fifoMap[key] = fifoName
            mlirBlock.fifoTypes[key] = tileTy

        # Declare external kernel — once per (kernel, device). The deployer
        # threads a shared set of already-declared kernel names so multiple
        # compute cores in the same device block don't redefine the symbol.
        argTypes = template.kernelArgTypes(tileTy)
        declared = getattr(mlirBlock, "declaredKernels", None)
        if declared is None or template.KERNEL_FN not in declared:
            aie_d.external_func(template.KERNEL_FN, argTypes, link_with = template.KERNEL_OBJ)
            if declared is not None:
                declared.add(template.KERNEL_FN)

        return ctxt, mlirBlock
