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
from Deeploy.Targets.XDNA2.Platform import NPU2_MEM_TILE_ROW, XDNA2AIECoreEngine, XDNA2MemTileDataMover, \
    XDNA2MemTileExecutionEngine, XDNA2ShimTileDataMover

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
        self._extractTargetCoreEngineToContext()
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

    def _extractTargetCoreEngineToContext(self) -> None:
        """Copy ``gs.Variable._targetCoreEngine`` into buffers.

        Set by :class:`XDNA2HybridElementwiseSpatialSplitPass` on chunk
        intermediates so the Distribute / Join codegen passes know which
        AIE compute tile each small mem-tile↔core FIFO connects to.
        """
        for tensor in self.graph.tensors().values():
            tgt = getattr(tensor, "_targetCoreEngine", None)
            if tgt is None:
                continue
            self.ctxt.lookup(tensor.name)._targetCoreEngine = tgt

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

        # Per-chunk per-tile size, derived from the consuming/producing
        # core node's tiling solution. The Distribute / Join passes need
        # this to size their small memtile↔core ObjectFifo elements
        # (== one kernel-acquire object), independently of how big the
        # full chunk is. Without this lookup, the small FIFO would carry
        # the entire chunk per object and the consuming core's
        # `objectfifo.acquire` (sized to one tile) would fail MLIR
        # verification with "ObjectFifo element and ObjectFifoSubview
        # element must match".
        chunkTileElems = self._buildChunkTileElems(nodes)

        # Memtile-engine nodes (Split / Concat) MUST run their device passes
        # before core-engine nodes — they populate the deployer-shared
        # ``fifoRegistry`` with memtile↔core ObjectFifo names that the
        # core nodes' MLIRObjectFifoPass consults to skip emitting its
        # own shim↔core FIFOs.
        nodes_ordered = sorted(nodes, key = lambda n: 0 if n["engineKind"] == "memtile" else 1)

        with mlir_mod_ctx() as ctx:

            @aie_d.device(aie_d.AIEDevice.npu2)
            def _device():
                coreTileMap = self._buildTileMap()       # AIE_c{c}r{r} → tile
                memTileMap = self._buildMemTileMap()     # MEM_c{c}    → tile
                shimTiles = self._buildShimTileMap()     # col         → tile

                # Single registry shared across every MLIRExecutionBlock in
                # this device. Distribute / Join passes write into it;
                # MLIRObjectFifoPass on compute nodes reads from it.
                fifoRegistry: Dict[Tuple[str, int, int], str] = {}

                # Track external_func declarations so multiple compute cores
                # don't redefine the same kernel symbol in this device block.
                declaredKernels = set()

                computeBlocks = []
                for node in nodes_ordered:
                    engineName = node["engineName"]
                    engineKind = node["engineKind"]
                    if engineKind == "core":
                        assert engineName in coreTileMap, (
                            f"Node '{node['nodeName']}' is colored '{engineName}' but no "
                            f"XDNA2AIECoreEngine with that name is registered.")
                        executionTile = coreTileMap[engineName]
                    else:  # memtile
                        assert engineName in memTileMap, (
                            f"Node '{node['nodeName']}' is colored '{engineName}' but no "
                            f"XDNA2MemTileExecutionEngine with that name is registered.")
                        executionTile = memTileMap[engineName]
                    # Representative shim for this column — used by both the
                    # compute path (shim↔core FIFOs in the legacy direct mode)
                    # and the memtile path (the big shim↔memtile FIFO).
                    shimTile = shimTiles[node["shimColPerKey"]["__representative__"]]
                    eb = MLIRExecutionBlock(computeTile = executionTile, shimTile = shimTile)
                    eb.operatorRepresentation = node["opRepr"]
                    eb.patternMemoryConstraint = node["tilingConstraint"]
                    eb.template = node["template"]
                    eb.declaredKernels = declaredKernels
                    # Plumb the per-block context the memtile-aware passes
                    # need: tile coords, the shared FIFO registry, and the
                    # per-engine tile lookup so Distribute/Join can resolve
                    # destination cores from chunk ``_targetCoreEngine``.
                    eb.tileCol = node["engine"].col
                    eb.tileRow = node["engine"].row
                    eb.fifoRegistry = fifoRegistry
                    eb.coreTileMap = coreTileMap
                    eb.chunkTileElems = chunkTileElems
                    if self.enableTrace:
                        eb.traceBufferSize = self.traceBufferSize

                    log.info(f"[XDNA2] Device phase ({engineKind}) for '{node['nodeName']}' "
                             f"on {engineName}")

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
            if isinstance(engine, XDNA2AIECoreEngine):
                engineKind = "core"
            elif isinstance(engine, XDNA2MemTileExecutionEngine):
                engineKind = "memtile"
            else:
                raise AssertionError(
                    f"Node '{nodeName}' is colored '{engineName}' which is neither an "
                    f"XDNA2AIECoreEngine nor an XDNA2MemTileExecutionEngine.")

            # Trace is only meaningful for compute cores. Skip the trace
            # pass injection for memtile-engine nodes — they have no
            # @aie_d.core block to trace.
            traceThis = (engineKind == "core" and self.enableTrace
                         and (not self.tracedEngines or engineName in self.tracedEngines))
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
                "engineKind": engineKind,
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

    def _buildChunkTileElems(self, nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        """For every chunk tensor referenced by a core node, look up the
        per-tile element count from that node's tiling solution.

        The Distribute / Join passes use this to size the small
        memtile↔core ObjectFifo objects (one tile per acquire). For
        elementwise sub-ops the tile shape is the same for every port
        because input and output L1 footprints match, so we record the
        first hit per tensor name and assume that's authoritative.
        """
        from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRObjectFifoPass import \
            _deriveTileShape   # noqa: E402

        out: Dict[str, int] = {}
        for node in nodes:
            if node["engineKind"] != "core":
                continue
            constraint = node["tilingConstraint"]
            if constraint is None:
                continue
            try:
                tileShape = _deriveTileShape(numElements = 0, patternMemoryConstraint = constraint)
            except Exception:
                continue
            tileElems = int(np.prod(tileShape))
            for key in list(node["template"].INPUT_KEYS) + list(node["template"].OUTPUT_KEYS):
                tensorName = node["opRepr"].get(key)
                if tensorName is None:
                    continue
                out.setdefault(tensorName, tileElems)
        return out

    def _buildMemTileMap(self) -> Dict[str, Any]:
        """One ``aie_d.tile`` per ``XDNA2MemTileExecutionEngine`` on the platform.

        Empty when the platform exposes none — that's the legacy
        shim-direct configuration and not an error.
        """
        engines = [e for e in self.Platform.engines if isinstance(e, XDNA2MemTileExecutionEngine)]
        return {engine.name: aie_d.tile(engine.col, NPU2_MEM_TILE_ROW) for engine in engines}

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

        Two cases per port:

        * Tensor IS in ``chunkToArg`` (it's a graph-IO chunk that maps
          to a runtime_sequence arg) — populate argIndexMap / argOffsets /
          transferLengths / shimColPerKey for that port. The runtime-
          sequence pass emits a shim DMA descriptor against the arg.

        * Tensor is NOT in ``chunkToArg`` (it's a chunk intermediate
          owned by a Split or Concat node) — set ``argIndexMap[key] = None``
          so the runtime-sequence pass skips this port. Data movement
          for it is fully described by the
          ``aie.objectfifo.link`` op the memtile-engine node emits.
        """
        gsNode: gs.Node = node["node"]
        template = node["template"]
        opRepr = node["opRepr"]

        argIndexMap: Dict[str, Optional[int]] = {}
        argOffsets: Dict[str, int] = {}
        transferLengths: Dict[str, int] = {}
        shimColPerKey: Dict[str, int] = {}

        for key in list(template.INPUT_KEYS) + list(template.OUTPUT_KEYS):
            tensorName = opRepr[key]
            if tensorName not in chunkToArg:
                # Chunk intermediate (between Split/Concat and a sub-op).
                # No shim DMA for this port; the link op moves the bytes.
                argIndexMap[key] = None
                continue
            argIdx, offset, length = chunkToArg[tensorName]
            argIndexMap[key] = argIdx
            argOffsets[key] = offset
            transferLengths[key] = length

            buf = self.ctxt.lookup(tensorName)
            engineName = getattr(buf, "_dataMoverEngine", None)
            assert engineName is not None, (
                f"Tensor '{tensorName}' (port '{key}' of '{gsNode.name}') has no _dataMoverEngine "
                f"annotation. The default-IO pass + spatial-split pass should have set it.")
            mover = self.Platform.getDataMoverEngine(engineName)
            assert isinstance(mover, (XDNA2ShimTileDataMover, XDNA2MemTileDataMover)), (
                f"Tensor '{tensorName}' is annotated with data mover '{engineName}', which is "
                f"neither a XDNA2ShimTileDataMover nor a XDNA2MemTileDataMover.")
            shimColPerKey[key] = mover.col

        # Pick a representative shim column for the node's FIFO declaration
        # — same column as the compute core / mem tile that owns it.
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
