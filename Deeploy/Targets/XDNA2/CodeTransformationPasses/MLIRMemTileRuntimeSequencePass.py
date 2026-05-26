# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Runtime-sequence pass for memtile-engine nodes (Split / Concat).

The actual host-visible DMA descriptor still targets the Shim, the
memtile half of the data path is fully described by the
``aie.objectfifo.link`` op emitted at device phase, and needs no host
configuration. This pass emits exactly one
``aiex_d.dma_configure_task_for(<big_fifo>)`` per memtile node:

* Split: input task on the big ``shim → memtile`` FIFO (no token).
* Concat: output task on the big ``memtile → shim`` FIFO
  (``issue_token = True``).

The task SSA value is appended to ``mlirBlock.issuedInputTasks`` /
``issuedOutputTasks`` so the deployer's batched await/free phase covers
it identically to compute-node tasks.

The Distribute / Join passes set ``mlirBlock.fifoMap['data_in']`` /
``['data_out']`` to the big-FIFO name. The deployer populates
``mlirBlock.argIndexMap['data_in'/'data_out']`` with the index of the
logical I/O in the runtime_sequence arg list, plus offsets and lengths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import aie.ir as ir
from aie.dialects import aie as aie_d
from aie.dialects import aiex as aiex_d

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext


class MLIRMemTileRuntimeSequencePass(MLIRCodeTransformationPass):

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        seqArgs = mlirBlock.runtimeSequenceArgs
        argIndexMap = getattr(mlirBlock, "argIndexMap", None) or {}
        argOffsets = getattr(mlirBlock, "argOffsets", {}) or {}
        transferLengths = getattr(mlirBlock, "transferLengths", {}) or {}

        # Lazily initialise the shared task lists (the deployer's batched
        # await/free phase walks these — same convention as MLIRRuntimeSequencePass).
        if not hasattr(mlirBlock, "issuedInputTasks"):
            mlirBlock.issuedInputTasks = []
        if not hasattr(mlirBlock, "issuedOutputTasks"):
            mlirBlock.issuedOutputTasks = []

        # We dispatch on which port the Distribute/Join pass populated.
        # Split owns 'data_in', Concat owns 'data_out'.
        if 'data_in' in mlirBlock.fifoMap:
            self._emit(mlirBlock, key = 'data_in', isOutput = False,
                       seqArgs = seqArgs, argIndexMap = argIndexMap,
                       argOffsets = argOffsets, transferLengths = transferLengths)
        if 'data_out' in mlirBlock.fifoMap:
            self._emit(mlirBlock, key = 'data_out', isOutput = True,
                       seqArgs = seqArgs, argIndexMap = argIndexMap,
                       argOffsets = argOffsets, transferLengths = transferLengths)

        return ctxt, mlirBlock

    @staticmethod
    def _emit(mlirBlock, key, isOutput, seqArgs, argIndexMap, argOffsets, transferLengths):
        argIdx = argIndexMap.get(key)
        if argIdx is None:
            # Deployer didn't bind this port to a runtime-sequence arg ->
            # nothing to emit. Shouldn't happen for memtile nodes whose
            # logical port is graph IO, but be defensive.
            return

        fifoName = mlirBlock.fifoMap[key]
        seqArg = seqArgs[argIdx]
        offset = int(argOffsets.get(key, 0))
        length = int(transferLengths.get(key, 0))
        assert length > 0, (
            f"Memtile runtime pass: transfer length for '{key}' is 0 — "
            "deployer must populate transferLengths for this port.")

        dims = [
            aie_d.bd_dim_layout(size = 1, stride = 0),
            aie_d.bd_dim_layout(size = 1, stride = 0),
            aie_d.bd_dim_layout(size = 1, stride = 0),
            aie_d.bd_dim_layout(size = length, stride = 1),
        ]

        if isOutput:
            task = aiex_d.dma_configure_task_for(fifoName, issue_token = True)
        else:
            task = aiex_d.dma_configure_task_for(fifoName)
        block = task.body.blocks.append()
        with ir.InsertionPoint(block):
            aie_d.dma_bd(seqArg, offset = offset, len = length, dimensions = dims, burst_length = 0)
            aie_d.end()
        aiex_d.dma_start_task(task)

        if isOutput:
            mlirBlock.issuedOutputTasks.append(task)
        else:
            mlirBlock.issuedInputTasks.append(task)
