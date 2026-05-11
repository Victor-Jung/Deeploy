# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from Deeploy.DeeployTypes import ConstantBuffer, DataMoverEngine, DeploymentEngine, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryPlatform
from Deeploy.Targets.Generic.Layers import AddLayer, LayerNormLayer, SiLULayer
from Deeploy.Targets.Generic.Parsers import AddParser, LayerNormParser, SiLUParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate, FreeTemplate
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
}

# Buffer classes reuse Generic templates since XDNA2Deployer manages its own
# output format (MLIR + test headers) and these templates are never rendered.


class XDNA2VariableBuffer(VariableBuffer):
    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate

    # None means "no transfer" (e.g. transient buffers that never cross a memory level).
    _dataMoverEngine: Optional[str] = None


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

    def getDataMoverEngine(self, name: str) -> Optional[DataMoverEngine]:
        return self._dataMoverByName.get(name)
