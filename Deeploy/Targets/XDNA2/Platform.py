# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from Deeploy.DeeployTypes import ConstantBuffer, DataMoverEngine, DeploymentEngine, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.EngineExtension.EngineAwareLayer import makeMappingEngineAware
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryPlatform
from Deeploy.Targets.Generic.Layers import AddLayer, ConcatLayer, GELULayer, LayerNormLayer, MulLayer, ReluLayer, \
    SiLULayer, SplitLayer, TanhLayer
from Deeploy.Targets.Generic.Parsers import AddParser, ConcatParser, GELUParser, LayerNormParser, ReluParser, \
    SiLUParser, SplitParser, TanhParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.XDNA2.Tiler import XDNA2AddTilingReadyBindings, XDNA2ConcatMemTileTilingReadyBindings, \
    XDNA2GeluTilingReadyBindings, XDNA2LayerNormTilingReadyBindings, XDNA2MulTilingReadyBindings, \
    XDNA2ReluTilingReadyBindings, XDNA2SiLUTilingReadyBindings, XDNA2SplitMemTileTilingReadyBindings, \
    XDNA2TanhTilingReadyBindings


class XDNA2GELULayer(GELULayer):
    """GELU with tanh-approximation op count matching the AIE kernel."""

    def computeOps(self):
        return self.mapper.parser.operatorRepresentation['size'] * 9


XDNA2AddMapper = NodeMapper(AddParser(), XDNA2AddTilingReadyBindings)
XDNA2MulMapper = NodeMapper(AddParser(), XDNA2MulTilingReadyBindings)
XDNA2GeluMapper = NodeMapper(GELUParser(), XDNA2GeluTilingReadyBindings)
XDNA2ReluMapper = NodeMapper(ReluParser(), XDNA2ReluTilingReadyBindings)
XDNA2SiLUMapper = NodeMapper(SiLUParser(), XDNA2SiLUTilingReadyBindings)
XDNA2TanhMapper = NodeMapper(TanhParser(), XDNA2TanhTilingReadyBindings)
XDNA2LayerNormMapper = NodeMapper(LayerNormParser(), XDNA2LayerNormTilingReadyBindings)
XDNA2SplitMemTileMapper = NodeMapper(SplitParser(), XDNA2SplitMemTileTilingReadyBindings)
XDNA2ConcatMemTileMapper = NodeMapper(ConcatParser(), XDNA2ConcatMemTileTilingReadyBindings)

XDNA2Mapping = makeMappingEngineAware({
    'Add': AddLayer([XDNA2AddMapper]),
    'Mul': MulLayer([XDNA2MulMapper]),
    'Gelu': XDNA2GELULayer([XDNA2GeluMapper]),
    'Relu': ReluLayer([XDNA2ReluMapper]),
    'Silu': SiLULayer([XDNA2SiLUMapper]),
    'Tanh': TanhLayer([XDNA2TanhMapper]),
    'LayerNormalization': LayerNormLayer([XDNA2LayerNormMapper]),
})

XDNA2MemTileMapping = makeMappingEngineAware({
    'Split': SplitLayer([XDNA2SplitMemTileMapper]),
    'Concat': ConcatLayer([XDNA2ConcatMemTileMapper]),
})

# Buffer classes reuse Generic templates since XDNA2Deployer manages its own
# output format (MLIR + test headers) and these templates are never rendered.


class XDNA2VariableBuffer(VariableBuffer):
    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate

    # None means "no transfer" (e.g. transient buffers that never cross a memory level).
    _dataMoverEngine: Optional[str] = None

    # Chunk-of-logical-parent metadata, set by XDNA2HybridElementwiseSpatialSplitPass on the
    # per-core chunk buffers. The logical parent is the L3-resident tensor
    # that the host actually allocates an XRT bo for; each chunk's DMA
    # descriptor accesses that bo at ``_chunkOffset`` element offset for
    # ``self.shape.numel()`` elements.
    #
    # ``_logicalParent = None`` means "this buffer IS the logical parent"
    _logicalParent: Optional[str] = None
    _chunkOffset: int = 0


class XDNA2TransientBuffer(TransientBuffer):
    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class XDNA2ConstantBuffer(ConstantBuffer):
    initTemplate = AllocateTemplate.referenceGlobalInitTemplate
    allocTemplate = AllocateTemplate.referenceGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.referenceGlobalTemplate

    _dataMoverEngine: Optional[str] = None


class XDNA2StructBuffer(StructBuffer):
    initTemplate = AllocateTemplate.referenceStructInitTemplate
    allocTemplate = AllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


XDNA2Optimizer = TopologyOptimizer([], name = "XDNA2Optimizer")

# ---------------------------------------------------------------------------
# XDNA2 hardware shape
# ---------------------------------------------------------------------------
NPU2_NUM_COLS = 8  # total columns in the AIE array
NPU2_NUM_AIE_ROWS = 4  # AIE compute cores per column (rows 2..5)
NPU2_AIE_ROW_OFFSET = 2  # first AIE row index
NPU2_MEM_TILE_ROW = 1  # mem tile row
NPU2_SHIM_TILE_ROW = 0  # shim tile row

VECTOR_WIDTH_BF16 = 16


def next_multiple(n: int, divisor: int) -> int:
    """Smallest multiple of ``divisor`` that is >= ``n``. ``divisor`` must be > 0."""
    assert divisor > 0, "next_multiple: divisor must be positive"
    return ((n + divisor - 1) // divisor) * divisor


class XDNA2AIECoreEngine(DeploymentEngine):
    """One AIE compute core, identified by its physical (col, row) placement.

    Each instance corresponds to exactly one AIE tile. Multi-core deployments
    construct one engine per core; the engine's ``name`` is used as the value
    written into ``node.attrs["engine"]`` by ``EngineColoringPass`` and is the
    key into ``EngineColoringDeployer.engineDict``, so it must be unique.
    """

    def __init__(self,
                 col: int = 0,
                 row: int = NPU2_AIE_ROW_OFFSET,
                 name: Optional[str] = None,
                 Mapping = None,
                 initCode: str = "",
                 includeList = None,
                 preferredMemoryLevel: Optional[str] = None) -> None:
        if name is None:
            name = f"AIE_c{col}r{row}"
        if Mapping is None:
            Mapping = XDNA2Mapping
        if includeList is None:
            includeList = []
        super().__init__(name, Mapping, initCode, includeList)
        self.col = col
        self.row = row
        # Default to this tile's own L1 — every AIE compute tile has its
        # own local memory in the full hardware hierarchy.
        self.preferredMemoryLevel = preferredMemoryLevel or f"L1_c{col}r{row}"


class XDNA2MemTileExecutionEngine(DeploymentEngine):
    """Mem tile as an execution engine for layout transformations."""

    def __init__(self,
                 col: int,
                 name: Optional[str] = None,
                 Mapping = None,
                 initCode: str = "",
                 includeList = None,
                 preferredMemoryLevel: Optional[str] = None) -> None:
        if name is None:
            name = f"MEM_c{col}"
        if Mapping is None:
            Mapping = XDNA2MemTileMapping
        if includeList is None:
            includeList = []
        super().__init__(name, Mapping, initCode, includeList)
        self.col = col
        self.row = NPU2_MEM_TILE_ROW
        # The big buffer this engine owns lives in this column's L2.
        self.preferredMemoryLevel = preferredMemoryLevel or f"L2_c{col}"


class XDNA2ShimTileDataMover(DataMoverEngine):
    """A shim DMA engine, identified by its column (row=0 implied)."""

    def __init__(self, col: int, name: Optional[str] = None) -> None:
        if name is None:
            name = f"shim_c{col}"
        super().__init__(name)
        self.col = col


class XDNA2AIECoreDataMover(DataMoverEngine):
    """The DMA of an AIE PE."""

    def __init__(self, col: int, row: int, name: Optional[str] = None) -> None:
        if name is None:
            name = f"core_dm_c{col}r{row}"
        super().__init__(name)
        self.col = col
        self.row = row


class XDNA2MemTileDataMover(DataMoverEngine):
    """A mem-tile DMA engine, identified by its column (row=1 implied)."""

    def __init__(self, col: int, name: Optional[str] = None) -> None:
        if name is None:
            name = f"mem_c{col}"
        super().__init__(name)
        self.col = col
        self.row = NPU2_MEM_TILE_ROW


class MemoryXDNA2Platform(MemoryPlatform):
    """XDNA2 platform with memory hierarchy + data-mover engine support."""

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 engines,
                 dataMoverEngines: Optional[List[DataMoverEngine]] = None,
                 variableBuffer = XDNA2VariableBuffer,
                 constantBuffer = XDNA2ConstantBuffer,
                 structBuffer = XDNA2StructBuffer,
                 transientBuffer = XDNA2TransientBuffer) -> None:
        super().__init__(memoryHierarchy, defaultTargetMemoryLevel, engines, variableBuffer, constantBuffer,
                         structBuffer, transientBuffer)
        self.dataMoverEngines: List[DataMoverEngine] = list(dataMoverEngines or [])
        self._dataMoverByName = {dm.name: dm for dm in self.dataMoverEngines}
        self._engineByName = {e.name: e for e in engines}

    def getDataMoverEngine(self, name: str) -> Optional[DataMoverEngine]:
        return self._dataMoverByName.get(name)

    def getTargetMemoryLevel(self, node, tensorName: str, ctxt) -> str:
        """Per-(node, tensor) target memory level for the tiler."""
        engineName = node.attrs.get("engine")
        if engineName is not None:
            engine = self._engineByName.get(engineName)
            if isinstance(engine, XDNA2AIECoreEngine):
                return engine.preferredMemoryLevel
        return self.defaultTargetMemoryLevel.name
