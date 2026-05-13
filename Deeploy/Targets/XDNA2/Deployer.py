# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 deployer — generates mlir-aie MLIR using ``aie.dialects``.

Unlike other Deeploy deployers that generate C code via Mako templates,
this deployer constructs an ``mlir.ir.Module`` with AIE dialect operations
and returns the verified MLIR text.

MLIR generation is split into two phases orchestrated by
:class:`MLIRCodeTransformation`:

1. **Device phase** — inside ``@aie_d.device(npu2)``: for each compute
   node, run ``devicePasses`` (ObjectFifo creation, external-kernel
   declaration) then call ``template.emit()`` (compute core only).
2. **Runtime-sequence phase** — inside ``@aiex_d.runtime_sequence``:
   for each compute node, run ``runtimeSequencePasses`` (Shim DMA configurations).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import aie.ir as ir
import numpy as np
import onnx_graphsurgeon as gs
from aie.dialects import aie as aie_d
from aie.dialects import aiex as aiex_d
from aie.extras.context import mlir_mod_ctx

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.DeeployTypes import DeploymentPlatform, TopologyOptimizer
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.MLIRDataTypes import MLIRCodeTransformation, MLIRExecutionBlock, MLIRNodeTemplate
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRCoreTracePass import MLIRCoreTracePass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRMemTracePass import MLIRMemTracePass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRTraceRuntimePass import MLIRTraceRuntimePass
from Deeploy.Targets.XDNA2.Platform import XDNA2AIECoreEngine, XDNA2ShimTileDataMover

_SHIM_TILE_ROW = 0


class XDNA2Deployer(SignPropDeployer):
    """Deployer for the XDNA2 (AIE2p) platform."""

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = False,
                 deeployStateDir: str = "DeeployStateDir",
                 inputOffsets: Optional[Dict[str, int]] = None,
                 enableTrace: bool = False,
                 traceBufferSize: int = 8192,
                 tracedEngines: Optional[Set[str]] = None):
        super().__init__(
            graph,
            deploymentPlatform,
            inputTypes,
            loweringOptimizer,
            scheduler,
            name,
            default_channels_first = default_channels_first,
            deeployStateDir = deeployStateDir,
            inputOffsets = inputOffsets if inputOffsets is not None else {},
        )
        self.enableTrace = enableTrace
        self.traceBufferSize = traceBufferSize
        self.tracedEngines = tracedEngines

    # ------------------------------------------------------------------
    # frontEnd extension: add data-mover extraction and invariant checks
    # ------------------------------------------------------------------

    def frontEnd(self):
        """Extend the standard frontEnd with post-parse extraction steps."""
        super().frontEnd()
        self._extractDataMoverToContext()
        self._extractChunkMetadataToContext()
        self._checkDataMoverInvariants()

    def _extractDataMoverToContext(self) -> None:
        """Copy every ``gs.Variable._dataMoverEngine`` hint into its buffer."""
        for tensor in self.graph.tensors().values():
            engineName = getattr(tensor, "_dataMoverEngine", None)
            if engineName is None:
                continue
            self.ctxt.lookup(tensor.name)._dataMoverEngine = engineName

    def _extractChunkMetadataToContext(self) -> None:
        """Copy ``gs.Variable._logicalParent`` / ``_chunkOffset`` into buffers."""
        for tensor in self.graph.tensors().values():
            parent = getattr(tensor, "_logicalParent", None)
            if parent is None:
                continue
            buf = self.ctxt.lookup(tensor.name)
            buf._logicalParent = parent
            buf._chunkOffset = int(getattr(tensor, "_chunkOffset", 0))

    def _checkDataMoverInvariants(self) -> None:
        """Every tensor in the graph must declare a data mover after frontEnd.

        Constants, graph IO, and intermediate tensors all need to be coloured by a DataMoverEngine.
        """
        for tensor in self.graph.tensors().values():
            buf = self.ctxt.lookup(tensor.name)  # raises KeyError if missing
            assert getattr(buf, "_dataMoverEngine", None) is not None, (
                f"Graph tensor '{tensor.name}' has no _dataMoverEngine. Every tensor must "
                f"declare a data mover; either XDNA2DefaultDataMoverPass missed it or a "
                f"downstream pass needs to set one.")

    # ------------------------------------------------------------------
    # MLIR generation
    # ------------------------------------------------------------------

    def generateMLIR(self) -> str:
        assert self.prepared, "XDNA2Deployer.generateMLIR() called before prepare()"

        nodes = self._collectNodes()
        if not nodes:
            raise RuntimeError("No compute nodes found — cannot generate MLIR.")

        # Recover logical I/O grouping from chunk names. The runtime sequence
        # is keyed off LOGICAL parents (one arg per logical input/output),
        # not chunks — keeps the kernel ABI to a small, fixed slot count.
        logicalNames, logicalLengths, chunkToArg = self._buildLogicalGrouping()

        # Resolve per-(compute node, port) DMA params: argIndex into the
        # logical-arg list, offset within that arg, transfer length.
        for node in nodes:
            self._resolveDmaPlacement(node, chunkToArg)

        with mlir_mod_ctx() as ctx:

            @aie_d.device(aie_d.AIEDevice.npu2)
            def _device():
                tileMap = self._buildTileMap()
                shimTiles = self._buildShimTileMap()

                # Track external_func declarations so multiple compute cores
                # don't redefine the same kernel symbol in this device block.
                declaredKernels = set()

                computeBlocks = []
                for node in nodes:
                    engineName = node["engineName"]
                    assert engineName in tileMap, (
                        f"Node '{node['nodeName']}' is colored '{engineName}' but no XDNA2AIECoreEngine "
                        f"with that name is registered on the platform.")
                    computeTile = tileMap[engineName]
                    # Pick the shim of the column the compute lives in. The
                    # data mover annotation has already established that this
                    # is the right shim for each port; we still need ONE
                    # representative shim tile for the FIFO declaration.
                    shimTile = shimTiles[node["shimColPerKey"]["__representative__"]]
                    eb = MLIRExecutionBlock(computeTile = computeTile, shimTile = shimTile)
                    eb.operatorRepresentation = node["opRepr"]
                    eb.patternMemoryConstraint = node["tilingConstraint"]
                    eb.template = node["template"]
                    eb.declaredKernels = declaredKernels
                    if self.enableTrace:
                        eb.traceBufferSize = self.traceBufferSize

                    log.info(f"[XDNA2] Device phase for '{node['nodeName']}' on {engineName}")

                    self.ctxt, eb = node["codeTransformer"].applyDevicePasses(self.ctxt, eb, node["nodeName"])
                    computeBlocks.append((node, eb))

                # === Runtime-sequence phase ===
                # Args are LOGICAL parents — one memref per logical I/O. The
                # spatial chunk count is hidden from the kernel ABI; the npu
                # instruction stream encodes the per-chunk DMA descriptors
                # against these logical buffers at the appropriate offsets.
                elemTy = ir.BF16Type.get()
                seqArgTypes = [ir.MemRefType.get((logicalLengths[name],), elemTy) for name in logicalNames]

                @aiex_d.runtime_sequence(*seqArgTypes)
                def _seq(*args):
                    # Phase 1: configure + start every node's DMAs back-to-back.
                    # The runtime-sequence passes append task SSA values to
                    # eb.issuedInputTasks / eb.issuedOutputTasks but do NOT
                    # emit awaits or frees themselves — that's deferred so all
                    # columns can run concurrently rather than the host blocking
                    # on each node's output before issuing the next node's input.
                    for node, eb in computeBlocks:
                        eb.runtimeSequenceArgs = list(args)
                        eb.argIndexMap = node["argIndexMap"]
                        eb.argOffsets = node["argOffsets"]
                        eb.transferLengths = node["transferLengths"]
                        log.info(f"[XDNA2] Runtime-sequence phase for '{node['nodeName']}' "
                                 f"(args={node['argIndexMap']}, lengths={node['transferLengths']})")
                        self.ctxt, eb = node["codeTransformer"].applyRuntimeSequencePasses(
                            self.ctxt, eb, node["nodeName"])

                    # Phase 2: await every output task across all nodes.
                    for _, eb in computeBlocks:
                        for task in getattr(eb, "issuedOutputTasks", []):
                            aiex_d.dma_await_task(task)

                    # Phase 3: free every input task across all nodes.
                    for _, eb in computeBlocks:
                        for task in getattr(eb, "issuedInputTasks", []):
                            aiex_d.dma_free_task(task)

            module = ctx.module
            assert module.operation.verify(), "[XDNA2] Generated MLIR module failed verification"

        mlirStr = str(module)
        log.info(f"[XDNA2] MLIR module generated ({len(mlirStr)} bytes)")
        return mlirStr

    # ------------------------------------------------------------------
    # node collection
    # ------------------------------------------------------------------

    def _collectNodes(self) -> List[Dict[str, Any]]:
        """Collect bound compute layers. Every node is a compute node now —
        data movement is annotation, not graph nodes."""
        engineByName = {e.name: e for e in self.Platform.engines}
        nodes = []
        for nodeName, layer in self.layerBinding.items():
            mapper = layer.mapper
            binder = mapper.binder
            template = binder.template
            opRepr = mapper.parser.operatorRepresentation
            codeTransformer = binder.codeTransformer
            tilingConstraint = getattr(binder.executionBlock, "patternMemoryConstraint", None)

            if not isinstance(template, MLIRNodeTemplate):
                raise RuntimeError(f"Node '{nodeName}' has no MLIRNodeTemplate — got {type(template).__name__}.")
            if not isinstance(codeTransformer, MLIRCodeTransformation):
                raise RuntimeError(
                    f"Node '{nodeName}' uses a non-MLIR CodeTransformation — got {type(codeTransformer).__name__}.")

            engineName = layer.node.attrs.get("engine")
            engine = engineByName.get(engineName) if engineName else None
            assert isinstance(engine, XDNA2AIECoreEngine), (
                f"Node '{nodeName}' is colored '{engineName}' which is not an XDNA2AIECoreEngine.")

            traceThis = self.enableTrace and (
                not self.tracedEngines or engineName in self.tracedEngines)
            if traceThis:
                codeTransformer = copy.copy(codeTransformer)
                codeTransformer.devicePasses = list(codeTransformer.devicePasses) + [
                    MLIRCoreTracePass(),
                    # MLIRMemTracePass(),
                ]
                codeTransformer.runtimeSequencePasses = [MLIRTraceRuntimePass()] + list(
                    codeTransformer.runtimeSequencePasses)

            nodes.append({
                "nodeName": nodeName,
                "node": layer.node,
                "op": layer.node.op,
                "template": template,
                "opRepr": opRepr,
                "codeTransformer": codeTransformer,
                "tilingConstraint": tilingConstraint,
                "engineName": engineName,
                "engine": engine,
            })
        return nodes

    def _buildTileMap(self) -> Dict[str, Any]:
        """One ``aie_d.tile`` per ``XDNA2AIECoreEngine`` declared by the platform."""
        engines = [e for e in self.Platform.engines if isinstance(e, XDNA2AIECoreEngine)]
        assert engines, ("XDNA2 platform exposes no XDNA2AIECoreEngine — at least one is required "
                         "for MLIR generation.")
        return {engine.name: aie_d.tile(engine.col, engine.row) for engine in engines}

    def _buildShimTileMap(self) -> Dict[int, Any]:
        """One ``aie_d.tile`` per shim column referenced on the platform."""
        shimDataMovers = [dm for dm in getattr(self.Platform, "dataMoverEngines", [])
                          if isinstance(dm, XDNA2ShimTileDataMover)]
        assert shimDataMovers, "XDNA2 platform exposes no XDNA2ShimTileDataMover."
        return {dm.col: aie_d.tile(dm.col, _SHIM_TILE_ROW) for dm in shimDataMovers}

    @staticmethod
    def _mlirElemType(buf) -> Any:
        """Map a Deeploy buffer's declared element type to an MLIR type."""
        # The XDNA2 path is BF16-only today; fall back to BF16 for unknown.
        return ir.BF16Type.get()

    # ------------------------------------------------------------------
    # logical I/O grouping (chunk → logical-arg, offset, length)
    # ------------------------------------------------------------------

    def _buildLogicalGrouping(self) -> Tuple[List[str], Dict[str, int], Dict[str, Tuple[int, int, int]]]:
        """Recover logical I/O grouping from chunked graph inputs/outputs."""
        input_groups = self._logicalGroups([t.name for t in self.graph.inputs])
        output_groups = self._logicalGroups([t.name for t in self.graph.outputs])

        # Logical args appear in declaration order: inputs first, then outputs.
        logicalNames = list(input_groups.keys()) + list(output_groups.keys())
        all_groups = {**input_groups, **output_groups}

        chunkToArg: Dict[str, Tuple[int, int, int]] = {}
        logicalLengths: Dict[str, int] = {}
        for argIdx, parent in enumerate(logicalNames):
            total = 0
            for offset, chunkName in all_groups[parent]:
                buf = self.ctxt.lookup(chunkName)
                length = int(np.prod(_safe_shape(buf)))
                chunkToArg[chunkName] = (argIdx, offset, length)
                total = max(total, offset + length)
            logicalLengths[parent] = total

        return logicalNames, logicalLengths, chunkToArg

    def _logicalGroups(self, names: List[str]) -> Dict[str, List[Tuple[int, str]]]:
        """Group ``names`` by their buffer-side ``_logicalParent`` field.

        Returns ``{parent_name: [(offset, chunk_name), ...]}`` with each
        group sorted by offset. Buffers whose ``_logicalParent`` is None
        are their own one-entry group at offset 0.
        """
        groups: Dict[str, List[Tuple[int, str]]] = {}
        for n in names:
            buf = self.ctxt.lookup(n)
            parent = getattr(buf, "_logicalParent", None) or n
            offset = int(getattr(buf, "_chunkOffset", 0))
            groups.setdefault(parent, []).append((offset, n))
        for parent in groups:
            groups[parent].sort(key = lambda p: p[0])
        return groups

    # ------------------------------------------------------------------
    # buffer-anchored DMA placement
    # ------------------------------------------------------------------

    def _resolveDmaPlacement(self, node: Dict[str, Any],
                             chunkToArg: Dict[str, Tuple[int, int, int]]) -> None:
        """Populate per-key DMA params from buffers' ``_dataMoverEngine`` and
        the chunk → (logical arg, offset, length) map.

        Each template port (INPUT_KEYS / OUTPUT_KEYS) maps via opRepr to a
        tensor that is, after spatial split, a chunk graph input/output. We
        look its row up in ``chunkToArg`` to learn which runtime_sequence
        arg it lives in and at what offset.
        """
        gsNode: gs.Node = node["node"]
        template = node["template"]
        opRepr = node["opRepr"]

        argIndexMap: Dict[str, int] = {}
        argOffsets: Dict[str, int] = {}
        transferLengths: Dict[str, int] = {}
        shimColPerKey: Dict[str, int] = {}

        for key in list(template.INPUT_KEYS) + list(template.OUTPUT_KEYS):
            tensorName = opRepr[key]
            assert tensorName in chunkToArg, (
                f"Node '{gsNode.name}' references tensor '{tensorName}' (port '{key}') which is "
                f"not in the logical-arg map — the spatial split pass should have promoted all "
                f"compute IO to graph IO.")
            argIdx, offset, length = chunkToArg[tensorName]
            argIndexMap[key] = argIdx
            argOffsets[key] = offset
            transferLengths[key] = length

            buf = self.ctxt.lookup(tensorName)
            engineName = getattr(buf, "_dataMoverEngine", None)
            assert engineName is not None, (
                f"Tensor '{tensorName}' (port '{key}' of '{gsNode.name}') has no _dataMoverEngine "
                f"annotation. XDNA2ElementwiseSpatialSplitPass + XDNA2AnnotateDataMoverPass should have set it.")
            mover = self.Platform.getDataMoverEngine(engineName)
            assert isinstance(mover, XDNA2ShimTileDataMover), (
                f"Tensor '{tensorName}' is annotated with data mover '{engineName}', which is not a "
                f"XDNA2ShimTileDataMover.")
            shimColPerKey[key] = mover.col

        # Pick a representative shim column for the compute node's FIFO
        # declaration — same column as the compute core.
        shimColPerKey["__representative__"] = node["engine"].col

        node["argIndexMap"] = argIndexMap
        node["argOffsets"] = argOffsets
        node["transferLengths"] = transferLengths
        node["shimColPerKey"] = shimColPerKey


def _safe_shape(buf) -> tuple:
    """Tolerate buffers whose shape is stored as either int or sequence."""
    shape = buf.shape
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)
