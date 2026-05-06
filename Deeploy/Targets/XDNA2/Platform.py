# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NetworkContext, NodeMapper, \
    NodeTemplate, StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryPlatform, MemoryPlatformWrapper
from Deeploy.Targets.Generic.Layers import AddLayer, ConcatLayer, LayerNormLayer, SiLULayer
from Deeploy.Targets.Generic.Parsers import AddParser, LayerNormParser, SiLUParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.XDNA2.ShimNodes import ShimReadLayer, ShimWriteLayer, XDNA2ConcatMappers, XDNA2ShimReadMapper, \
    XDNA2ShimWriteMapper
from Deeploy.Targets.XDNA2.Tiler import XDNA2AddTilingReadyBindings, XDNA2LayerNormTilingReadyBindings, \
    XDNA2SiLUTilingReadyBindings

# XDNA2 always requires tiling (ObjectFifo streaming).
XDNA2AddMapper = NodeMapper(AddParser(), XDNA2AddTilingReadyBindings)
XDNA2SiLUMapper = NodeMapper(SiLUParser(), XDNA2SiLUTilingReadyBindings)
XDNA2LayerNormMapper = NodeMapper(LayerNormParser(), XDNA2LayerNormTilingReadyBindings)

XDNA2Mapping = {
    'Add': AddLayer([XDNA2AddMapper]),
    'Silu': SiLULayer([XDNA2SiLUMapper]),
    'LayerNormalization': LayerNormLayer([XDNA2LayerNormMapper]),
    # Shim-tile data movement; see ShimNodes.py.
    'ShimRead': ShimReadLayer([XDNA2ShimReadMapper]),
    'ShimWrite': ShimWriteLayer([XDNA2ShimWriteMapper]),
    # Concat marker
    'Concat': ConcatLayer(XDNA2ConcatMappers),
}

# Buffer classes reuse Generic templates since XDNA2Deployer manages its own
# output format (MLIR + test headers) and these templates are never rendered.


class XDNA2VariableBuffer(VariableBuffer):
    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class XDNA2TransientBuffer(TransientBuffer):
    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class XDNA2ConstantBuffer(ConstantBuffer):
    initTemplate = AllocateTemplate.referenceGlobalInitTemplate
    allocTemplate = AllocateTemplate.referenceGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.referenceGlobalTemplate


class XDNA2StructBuffer(StructBuffer):
    initTemplate = AllocateTemplate.referenceStructInitTemplate
    allocTemplate = AllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


XDNA2Optimizer = TopologyOptimizer([], name = "XDNA2Optimizer")


class XDNA2Engine(DeploymentEngine):

    def __init__(self, name: str = "XDNA2", Mapping = XDNA2Mapping, initCode: str = "", includeList = None) -> None:
        if includeList is None:
            includeList = []
        super().__init__(name, Mapping, initCode, includeList)


class XDNA2AIECoreEngine(DeploymentEngine):
    """One AIE compute core, identified by its physical (col, row) placement.

    Each instance corresponds to exactly one AIE tile. Multi-core deployments
    construct one engine per core; the engine's ``name`` is used as the value
    written into ``node.attrs["engine"]`` by ``EngineColoringPass`` and is the
    key into ``EngineColoringDeployer.engineDict``, so it must be unique.
    """

    def __init__(self,
                 col: int = 0,
                 row: int = 2,
                 name: Optional[str] = None,
                 Mapping = None,
                 initCode: str = "",
                 includeList = None,
                 preferredMemoryLevel: str = "L1") -> None:
        if name is None:
            name = f"AIE_c{col}r{row}"
        if Mapping is None:
            Mapping = XDNA2Mapping
        if includeList is None:
            includeList = []
        super().__init__(name, Mapping, initCode, includeList)
        self.col = col
        self.row = row
        self.preferredMemoryLevel = preferredMemoryLevel


class XDNA2ShimEngine(DeploymentEngine):
    """One AIE shim tile (row 0), identified by its column.

    A shim engine "executes" by issuing DMA descriptors that move data between
    L3 (DRAM) and L1 (compute-tile local memory) via ObjectFifos. Tensors
    produced or consumed by shim-colored nodes live in L3 — that fact falls
    out of ``preferredMemoryLevel = "L3"`` here, with no special-casing
    required in :class:`MemoryXDNA2Platform.getTargetMemoryLevel`.

    In the current implementation, shim-colored nodes (``ShimRead`` /
    ``ShimWrite``) carry the DMA parameters (offset, length, target tensor)
    as ONNX attributes but do not own MLIR emission — the surrounding compute
    node's runtime-sequence pass reads those attributes when emitting its
    per-compute DMA block. We will move emission ownership to shim nodes
    when we redesign DMA scheduling.
    """

    def __init__(self,
                 col: int = 0,
                 name: Optional[str] = None,
                 Mapping = None,
                 initCode: str = "",
                 includeList = None,
                 preferredMemoryLevel: str = "L3") -> None:
        if name is None:
            name = f"shim_c{col}"
        if Mapping is None:
            Mapping = XDNA2Mapping
        if includeList is None:
            includeList = []
        super().__init__(name, Mapping, initCode, includeList)
        self.col = col
        self.row = 0
        self.preferredMemoryLevel = preferredMemoryLevel


class XDNA2Platform(DeploymentPlatform):

    def __init__(self,
                 engines = None,
                 variableBuffer = XDNA2VariableBuffer,
                 constantBuffer = XDNA2ConstantBuffer,
                 structBuffer = XDNA2StructBuffer,
                 transientBuffer = XDNA2TransientBuffer):
        if engines is None:
            engines = [XDNA2Engine()]
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)


class MemoryXDNA2Platform(MemoryPlatform):
    """XDNA2 platform with memory hierarchy support for tiling.

    Defines the memory hierarchy:
    - L1: 64KB per AIE core (local memory)
    - L3: Shared memory for entire AIE array
    """

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 engines = None,
                 variableBuffer = XDNA2VariableBuffer,
                 constantBuffer = XDNA2ConstantBuffer,
                 structBuffer = XDNA2StructBuffer,
                 transientBuffer = XDNA2TransientBuffer) -> None:
        if engines is None:
            engines = [XDNA2AIECoreEngine()]
        super().__init__(memoryHierarchy, defaultTargetMemoryLevel, engines, variableBuffer, constantBuffer,
                         structBuffer, transientBuffer)
        self._engineByName = {e.name: e for e in engines}

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        """Resolve the preferred memory level via the node's engine color.

        ``EngineColoringPass`` writes the engine's name into
        ``node.attrs["engine"]``; we look that up to get the engine instance
        and return its ``preferredMemoryLevel``. Compute engines yield L1,
        shim engines yield L3 — no per-op special case needed.
        """
        engineName = node.attrs.get("engine")
        if engineName is not None:
            engine = self._engineByName.get(engineName)
            if isinstance(engine, (XDNA2AIECoreEngine, XDNA2ShimEngine)):
                return engine.preferredMemoryLevel
        return self.defaultTargetMemoryLevel.name


class MemoryXDNA2PlatformWrapper(MemoryPlatformWrapper):
    """Wrapper for XDNA2Platform with memory-level support."""

    def __init__(self, platform: XDNA2Platform, memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel):
        assert isinstance(platform, XDNA2Platform), \
            f"Given platform is not an instance of XDNA2Platform. Platform type: {type(platform).__name__}"
        super().__init__(platform, memoryHierarchy, defaultTargetMemoryLevel)
        self._engineByName = {e.name: e for e in platform.engines}

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        engineName = node.attrs.get("engine")
        if engineName is not None:
            engine = self._engineByName.get(engineName)
            if isinstance(engine, (XDNA2AIECoreEngine, XDNA2ShimEngine)):
                return engine.preferredMemoryLevel
        return self.defaultTargetMemoryLevel.name
