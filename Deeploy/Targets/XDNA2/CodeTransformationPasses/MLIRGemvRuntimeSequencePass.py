# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Runtime-sequence pass: shim DMA descriptors for BF16 GEMV (row-major kernel).

Because the generic kernel reads A row-major and reduces over the full K per
call, every transfer is a single plain contiguous (linear) shim DMA — the
ObjectFifo chunks each linear stream into the per-object tiles the core
acquires. This mirrors IRON's gemv runtime (one linear ``fill`` per input, one
linear ``drain`` for the output):

* **A** (input, [M,N] row-major): one linear transfer of ``M*N`` elements. The
  inA ObjectFifo (element ``(DIM_M, N)``) dispenses ``M/DIM_M`` A tiles in row
  order.
* **x** (input, [N,1]): one linear transfer of ``N`` elements. The inB
  ObjectFifo (element ``(N,)``) dispenses a single object, held by the core.
* **y** (output, [M,1]): one linear transfer of ``M`` elements, gathered from
  the ``M/DIM_M`` outC tiles the core produces.

Awaits/frees are emitted by the deployer's batched phase over the issued task
lists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import aie.ir as ir
from aie.dialects import aie as aie_d
from aie.dialects import aiex as aiex_d

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext


class MLIRGemvRuntimeSequencePass(MLIRCodeTransformationPass):

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        template = mlirBlock.template
        aKey, bKey = template.INPUT_KEYS
        (cKey,) = template.OUTPUT_KEYS

        opRepr = mlirBlock.operatorRepresentation
        M = int(opRepr['M'])
        N = int(opRepr['N'])  # reduction dim K

        seqArgs = mlirBlock.runtimeSequenceArgs
        argIndexMap = mlirBlock.argIndexMap

        if not hasattr(mlirBlock, "issuedInputTasks"):
            mlirBlock.issuedInputTasks = []
        if not hasattr(mlirBlock, "issuedOutputTasks"):
            mlirBlock.issuedOutputTasks = []

        def _emit(key: str, length: int, isOutput: bool) -> None:
            if argIndexMap.get(key) is None:
                return
            seqArg = seqArgs[argIndexMap[key]]
            # Single contiguous (linear) transfer; the ObjectFifo reshapes the
            # stream into per-object tiles.
            dims = [aie_d.bd_dim_layout(size = length, stride = 1)]
            fifoName = mlirBlock.fifoMap[key]
            if isOutput:
                task = aiex_d.dma_configure_task_for(fifoName, issue_token = True)
            else:
                task = aiex_d.dma_configure_task_for(fifoName)
            block = task.body.blocks.append()
            with ir.InsertionPoint(block):
                aie_d.dma_bd(seqArg, offset = 0, len = length, dimensions = dims, burst_length = 0)
                aie_d.end()
            aiex_d.dma_start_task(task)
            (mlirBlock.issuedOutputTasks if isOutput else mlirBlock.issuedInputTasks).append(task)

        _emit(aKey, M * N, False)  # A: full [M,N] matrix, linear
        _emit(bKey, N, False)      # x: full [N] vector, linear
        _emit(cKey, M, True)       # y: full [M] output, linear

        return ctxt, mlirBlock
