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
   for each compute node, run ``runtimeSequencePasses`` (DMA configuration).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

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
from Deeploy.Targets.XDNA2.Platform import XDNA2AIECoreEngine, XDNA2ShimEngine

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
                 traceBufferSize: int = 8192):
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

    # ------------------------------------------------------------------
    # MLIR generation
    # ------------------------------------------------------------------

    def generateMLIR(self) -> str:
        assert self.prepared, "XDNA2Deployer.generateMLIR() called before prepare()"

        nodes = self._collectNodes()
        if not any(n["isCompute"] for n in nodes):
            raise RuntimeError("No compute nodes found — cannot generate MLIR.")

        # The runtime_sequence args are the original graph inputs + outputs
        # in declaration order; the host harness writes them in the same
        # order from testinputs.h / testoutputs.h.
        graphArgNames = [t.name for t in self.graph.inputs] + [t.name for t in self.graph.outputs]
        graphArgIndex = {name: i for i, name in enumerate(graphArgNames)}

        # For every compute node, walk one hop to its surrounding ShimRead /
        # ShimWrite to gather DMA params. This is the only place the shim
        # nodes' ``offset`` / ``length`` attrs are consumed.
        for node in nodes:
            if not node["isCompute"]:
                continue
            self._resolveArgMapping(node, graphArgIndex)

        with mlir_mod_ctx() as ctx:

            @aie_d.device(aie_d.AIEDevice.npu2)
            def _device():
                tileMap = self._buildTileMap()
                shimMap = self._buildShimMap(tileMap)

                # === Device phase ===
                # Track external_func declarations so multiple compute cores
                # don't redefine the same kernel symbol in this device block.
                declaredKernels = set()

                computeBlocks = []
                for node in nodes:
                    if not node["isCompute"]:
                        log.info(f"[XDNA2] Skipping shim node '{node['nodeName']}' ({node['op']})")
                        continue

                    engineName = node["engineName"]
                    assert engineName is not None, (
                        f"Node '{node['nodeName']}' has no engine color — wrap the deployer with "
                        f"EngineColoringDeployerWrapper.")
                    assert engineName in tileMap, (
                        f"Node '{node['nodeName']}' is colored '{engineName}' but no XDNA2AIECoreEngine "
                        f"with that name is registered on the platform.")
                    computeTile = tileMap[engineName]
                    shimTile = shimMap[engineName]
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

                if not computeBlocks:
                    raise RuntimeError("Device phase produced no compute cores.")

                # === Runtime-sequence phase ===
                seqArgTypes = []
                for name in graphArgNames:
                    buf = self.ctxt.lookup(name)
                    elemTy = self._mlirElemType(buf)
                    n = int(np.prod(buf.shape))
                    seqArgTypes.append(ir.MemRefType.get((n,), elemTy))

                @aiex_d.runtime_sequence(*seqArgTypes)
                def _seq(*args):
                    for node, eb in computeBlocks:
                        eb.runtimeSequenceArgs = list(args)
                        eb.argIndexMap = node["argIndexMap"]
                        eb.argOffsets = node["argOffsets"]
                        eb.transferLengths = node["transferLengths"]
                        log.info(f"[XDNA2] Runtime-sequence phase for '{node['nodeName']}' "
                                 f"(offsets={node['argOffsets']})")
                        self.ctxt, eb = node["codeTransformer"].applyRuntimeSequencePasses(
                            self.ctxt, eb, node["nodeName"])

            module = ctx.module
            assert module.operation.verify(), "[XDNA2] Generated MLIR module failed verification"

        mlirStr = str(module)
        log.info(f"[XDNA2] MLIR module generated ({len(mlirStr)} bytes)")
        return mlirStr

    # ------------------------------------------------------------------
    # node collection / engine-kind dispatch
    # ------------------------------------------------------------------

    def _collectNodes(self) -> List[Dict[str, Any]]:
        """Collect bound layers and classify them by engine kind.

        ``isCompute`` is determined by the colored engine, not by a
        per-template flag: shim-engine-colored nodes (``ShimRead`` /
        ``ShimWrite``) are skipped in MLIR emission; everything else is a
        compute node.
        """
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

            engineName = layer.node.attrs.get("engine")
            engine = engineByName.get(engineName) if engineName else None
            isCompute = isinstance(engine, XDNA2AIECoreEngine)

            if isCompute and not isinstance(codeTransformer, MLIRCodeTransformation):
                raise RuntimeError(
                    f"Node '{nodeName}' uses a non-MLIR CodeTransformation — got {type(codeTransformer).__name__}.")

            if isCompute and self.enableTrace:
                codeTransformer = copy.copy(codeTransformer)
                codeTransformer.devicePasses = list(codeTransformer.devicePasses) + [
                    MLIRCoreTracePass(),
                    MLIRMemTracePass(),
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
                "isCompute": isCompute,
                "engineName": engineName,
            })
        return nodes

    def _buildTileMap(self) -> Dict[str, Any]:
        """One ``aie_d.tile`` per ``XDNA2AIECoreEngine`` declared by the platform."""
        engines = [e for e in self.Platform.engines if isinstance(e, XDNA2AIECoreEngine)]
        assert engines, ("XDNA2 platform exposes no XDNA2AIECoreEngine — at least one is required "
                         "for MLIR generation.")
        return {engine.name: aie_d.tile(engine.col, engine.row) for engine in engines}

    def _buildShimMap(self, tileMap: Dict[str, Any]) -> Dict[str, Any]:
        """One ``aie_d.tile`` shim per occupied compute column.

        NPU2 has one shim per column with limited DMA channels — funnelling
        several compute cores through a single shim trips the resource
        allocator. Pairing each compute column with its own shim avoids the
        saturation. We materialize the shim tile from the matching
        :class:`XDNA2ShimEngine` registered on the platform.
        """
        coreEngines = {e.name: e for e in self.Platform.engines if isinstance(e, XDNA2AIECoreEngine)}
        shimByCol = {e.col: e for e in self.Platform.engines if isinstance(e, XDNA2ShimEngine)}
        shimTiles: Dict[int, Any] = {}
        result: Dict[str, Any] = {}
        for engineName in tileMap:
            col = coreEngines[engineName].col
            assert col in shimByCol, (
                f"No XDNA2ShimEngine for column {col}; the platform must expose one shim engine per "
                f"AIE core column.")
            if col not in shimTiles:
                shimTiles[col] = aie_d.tile(col, shimByCol[col].row)
            result[engineName] = shimTiles[col]
        return result

    @staticmethod
    def _mlirElemType(buf) -> Any:
        """Map a Deeploy buffer's declared element type to an MLIR type."""
        # The XDNA2 path is BF16-only today; fall back to BF16 for unknown.
        return ir.BF16Type.get()

    # ------------------------------------------------------------------
    # Shim-anchored arg resolution
    # ------------------------------------------------------------------

    def _resolveArgMapping(self, node: Dict[str, Any], graphArgIndex: Dict[str, int]) -> None:
        """Populate ``argIndexMap`` / ``argOffsets`` / ``transferLengths`` for one compute node.

        Walks one hop in the graph: every input edge of a compute node has
        a ``ShimRead`` producer; every output edge has a ``ShimWrite``
        consumer. The shim node's ``(offset, length)`` attributes plus its
        partner L3 tensor's runtime-arg index give the DMA descriptor.
        """
        gsNode: gs.Node = node["node"]
        template = node["template"]
        opRepr = node["opRepr"]

        argIndexMap: Dict[str, int] = {}
        argOffsets: Dict[str, int] = {}
        transferLengths: Dict[str, int] = {}

        # Inputs: producer must be a ShimRead.
        for key, inp in zip(template.INPUT_KEYS, gsNode.inputs):
            shim = self._uniqueProducer(inp.name)
            assert shim is not None and shim.op == "ShimRead", (
                f"Compute node '{gsNode.name}' input '{inp.name}' is not produced by a ShimRead "
                f"(got {shim.op if shim else 'None'}). Every memory transition must be a shim node.")
            l3Name = shim.inputs[0].name
            assert l3Name in graphArgIndex, (
                f"ShimRead '{shim.name}' reads from '{l3Name}' which is not a graph input.")
            argIndexMap[key] = graphArgIndex[l3Name]
            argOffsets[key] = int(shim.attrs["offset"])
            transferLengths[key] = int(shim.attrs["length"])

        # Outputs: consumer (for THIS specific compute node) must be a ShimWrite.
        # Two graph shapes are valid:
        #   * num_chunks == 1: ShimWrite output IS the graph output tensor.
        #   * num_chunks  > 1: ShimWrite outputs a per-chunk intermediate; a
        #     downstream Concat marker reconstructs the graph output.
        for key, out in zip(template.OUTPUT_KEYS, gsNode.outputs):
            shim = self._uniqueConsumer(out.name)
            assert shim is not None and shim.op == "ShimWrite", (
                f"Compute node '{gsNode.name}' output '{out.name}' is not consumed by a ShimWrite "
                f"(got {shim.op if shim else 'None'}). Every memory transition must be a shim node.")
            shimOutName = shim.outputs[0].name
            if shimOutName in graphArgIndex:
                l3Name = shimOutName
            else:
                concat = self._uniqueConsumer(shimOutName)
                assert concat is not None and concat.op == "Concat", (
                    f"ShimWrite '{shim.name}' writes to '{shimOutName}', which is neither a graph "
                    f"output nor consumed by a Concat marker.")
                l3Name = concat.outputs[0].name
                assert l3Name in graphArgIndex, (
                    f"Concat '{concat.name}' output '{l3Name}' is not a graph output.")
            argIndexMap[key] = graphArgIndex[l3Name]
            argOffsets[key] = int(shim.attrs["offset"])
            transferLengths[key] = int(shim.attrs["length"])

        node["argIndexMap"] = argIndexMap
        node["argOffsets"] = argOffsets
        node["transferLengths"] = transferLengths

    def _uniqueProducer(self, tensorName: str) -> Optional[gs.Node]:
        """Return the single producer node of ``tensorName`` (None if none)."""
        producers = [n for n in self.graph.nodes if any(o.name == tensorName for o in n.outputs)]
        if not producers:
            return None
        assert len(producers) == 1, f"Tensor '{tensorName}' has multiple producers: {[p.name for p in producers]}"
        return producers[0]

    def _uniqueConsumer(self, tensorName: str) -> Optional[gs.Node]:
        """Return the single consumer of ``tensorName``.

        Compute-node output tensors flow into exactly one ``ShimWrite``
        (each compute node has its own dedicated ShimWrite even when several
        ShimWrites share a graph-output tensor downstream — the multi-producer
        case is on the graph-output side, not the compute-output side).
        """
        consumers = [n for n in self.graph.nodes if any(i.name == tensorName for i in n.inputs)]
        if not consumers:
            return None
        assert len(consumers) == 1, f"Tensor '{tensorName}' has multiple consumers: {[c.name for c in consumers]}"
        return consumers[0]
