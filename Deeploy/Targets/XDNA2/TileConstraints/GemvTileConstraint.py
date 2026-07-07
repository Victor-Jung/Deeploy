# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""GEMV (matrix-vector) tile constraint for XDNA2.

GEMV: ``y = A @ x`` with A=[M,N], x=[N,1], y=[M,1] (N is the reduction dim K).

The row-major ``mv.o`` kernel (IRON generic ABI) reduces over the **full** K in
one call and computes ``DIM_M`` output rows, so only the M dimension is tiled;
K is not. The M-tiling schedule (M/DIM_M output tiles) is driven by
:class:`MLIRGemvComputeCorePass`, so this constraint only pins the per-tile
geometry:

* A  tile  = (DIM_M, N)   — one output-row band, full reduction dim
* x  tile  = (N, 1)       — the whole vector, held on-core
* y  tile  = (DIM_M, 1)

and requires M % DIM_M == 0 and N % VEC_SIZE == 0 (the kernel's SIMD chunk must
divide K).

NOTE (iteration point): the full x vector and full K row band must fit in L1.
For large K this would require memtile-level K-tiling with an accumulate-into-C
kernel variant (the generic kernel overwrites C and does not accumulate); that
is future work. The single input cube per tensor below describes the L1 tile
shape the solver must budget for.
"""

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.XDNA2.TileConstraints.DivisibilityHelper import addDivisibilityConstraints
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme

# Output-row tile granularity (must match MLIRGemvObjectFifoPass.DIM_M).
DIM_M = 32
# SIMD chunk size of the vectorized kernel (must match mv.cc VEC_SIZE and
# divide K); K >= 2*VEC_SIZE is required by the kernel's pipelined loop.
VEC_SIZE = 16


class XDNA2GemvTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        aName = parseDict['A']  # [M, N]
        bName = parseDict['B']  # [N, 1]
        outName = parseDict['data_out']  # [M, 1]

        for bufferName in [aName, bName, outName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        aShape = ctxt.lookup(aName).shape
        assert len(aShape) == 2, f"GEMV expects 2-D A, got shape {aShape}"
        mLen, nLen = int(aShape[0]), int(aShape[1])
        assert mLen % DIM_M == 0, f"GEMV M={mLen} must be a multiple of DIM_M={DIM_M}"
        assert nLen % VEC_SIZE == 0, f"GEMV N(K)={nLen} must be a multiple of VEC_SIZE={VEC_SIZE}"
        assert nLen >= 2 * VEC_SIZE, f"GEMV N(K)={nLen} must be >= 2*VEC_SIZE={2 * VEC_SIZE} (kernel loop pipelining)"

        aM = tilerModel.getTensorDimVar(tensorName = aName, dimIdx = 0)
        aN = tilerModel.getTensorDimVar(tensorName = aName, dimIdx = 1)
        bN = tilerModel.getTensorDimVar(tensorName = bName, dimIdx = 0)
        bOne = tilerModel.getTensorDimVar(tensorName = bName, dimIdx = 1)
        outM = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 0)
        outOne = tilerModel.getTensorDimVar(tensorName = outName, dimIdx = 1)

        # Fixed per-tile geometry: M is tiled to DIM_M rows; K is held whole.
        tilerModel.addConstraint(aM == DIM_M)
        tilerModel.addConstraint(aN == nLen)
        tilerModel.addConstraint(bN == nLen)
        tilerModel.addConstraint(bOne == 1)
        tilerModel.addConstraint(outM == DIM_M)
        tilerModel.addConstraint(outOne == 1)

        # Geometry links: A rows == out rows; A cols == x rows (reduction).
        tilerModel.addConstraint(aM == outM)
        tilerModel.addConstraint(aN == bN)

        # XDNA2: no remainder tiles.
        addDivisibilityConstraints(tilerModel, outName, ctxt)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        addrNames = ['A', 'B', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        nLen = int(operatorRepresentation['N'])  # reduction dim K (held whole)

        replacements = {"M": [], "N": []}
        from Deeploy.AbstractDataTypes import PointerClass
        from Deeploy.CommonExtensions.DataTypes import uint16_t
        replacementTypes = {"M": PointerClass(uint16_t), "N": PointerClass(uint16_t)}

        inputLoadSchedule = []
        outputLoadSchedule = []

        # Each output cube is one [DIM_M, 1] y-tile. Its A slice is the full
        # [DIM_M, N] row band and its x slice is the whole [N, 1] vector; the
        # kernel reduces over all N in one call.
        for cube in outputCubes:
            mTile = int(cube.dims[0])
            replacements["M"].append(mTile)
            replacements["N"].append(nLen)

            mOffset = int(cube.offset[0]) if hasattr(cube, "offset") else 0
            aCube = HyperRectangle((mOffset, 0), (mTile, nLen))
            xCube = HyperRectangle((0, 0), (nLen, 1))
            inputLoadSchedule.append({"A": aCube, "B": xCube})
            outputLoadSchedule.append({"data_out": cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
