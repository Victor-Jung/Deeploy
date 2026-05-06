# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Runtime-sequence pass that configures shim DMA for L3 ↔ L1 transfers.

Given an :class:`MLIRExecutionBlock` whose device-phase passes have already
populated ``fifoMap``, ``numElements``, and ``runtimeSequenceArgs``, this
pass emits ``aiex_d.dma_configure_task_for`` / ``dma_start_task`` /
``dma_await_task`` / ``dma_free_task`` operations directly into the current
``@aiex_d.runtime_sequence`` insertion point.

The pass is operator-agnostic — it iterates over the FIFO map and
runtime-sequence arguments to configure DMA for every input and output
tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import aie.ir as ir
from aie.dialects import aie as aie_d
from aie.dialects import aiex as aiex_d

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext


class MLIRRuntimeSequencePass(MLIRCodeTransformationPass):
    """Emit DMA configuration inside a ``runtime_sequence`` block.

    All operator-specific metadata (tensor keys) is read from the
    template's ``INPUT_KEYS`` and ``OUTPUT_KEYS`` via ``mlirBlock.template``.
    """

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        template = mlirBlock.template
        inputTensorKeys = template.INPUT_KEYS
        outputTensorKeys = template.OUTPUT_KEYS

        # For a multi-core spatial split, every sub-Add reads/writes a
        # contiguous chunk of the (still single) graph-level tensor. The
        # deployer fills ``mlirBlock.argIndexMap`` with the runtime-sequence
        # arg index per tensor key and ``mlirBlock.argOffsets`` with the
        # per-key element offset into that arg. ``mlirBlock.transferLengths``
        # gives the number of elements transferred per key (defaults to
        # numElements for non-split nodes).
        numElements = mlirBlock.numElements
        seqArgs = mlirBlock.runtimeSequenceArgs
        argIndexMap = getattr(mlirBlock, "argIndexMap", None)
        argOffsets = getattr(mlirBlock, "argOffsets", {})
        transferLengths = getattr(mlirBlock, "transferLengths", {})

        inputTasks = []
        outputTasks = []

        allKeys = list(inputTensorKeys) + list(outputTensorKeys)
        for idx, key in enumerate(allKeys):
            fifoName = mlirBlock.fifoMap[key]
            isOutput = key in outputTensorKeys

            if argIndexMap is not None:
                seqArg = seqArgs[argIndexMap[key]]
            else:
                # Backwards compatible single-node path: positional mapping.
                seqArg = seqArgs[idx]

            offset = int(argOffsets.get(key, 0))
            length = int(transferLengths.get(key, numElements))

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
                outputTasks.append(task)
            else:
                inputTasks.append(task)

        # Await output tasks, then free input tasks
        for task in outputTasks:
            aiex_d.dma_await_task(task)
        for task in inputTasks:
            aiex_d.dma_free_task(task)

        return ctxt, mlirBlock
