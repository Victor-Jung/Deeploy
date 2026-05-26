# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 tiling constraints and tiling-ready node bindings for MLIR code generation."""

from Deeploy.Targets.XDNA2.Bindings import XDNA2AddBindings, XDNA2ConcatMemTileBindings, XDNA2GeluBindings, \
    XDNA2LayerNormBindings, XDNA2MulBindings, XDNA2ReluBindings, XDNA2SiLUBindings, XDNA2SplitMemTileBindings, \
    XDNA2TanhBindings
from Deeploy.Targets.XDNA2.TileConstraints.AddTileConstraint import XDNA2AddTileConstraint
from Deeploy.Targets.XDNA2.TileConstraints.LayerNormTileConstraint import XDNA2LayerNormTileConstraint
from Deeploy.Targets.XDNA2.TileConstraints.MemTileLayoutTileConstraint import XDNA2MemTileLayoutTileConstraint
from Deeploy.Targets.XDNA2.TileConstraints.UnaryTileConstraint import XDNA2UnaryTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

XDNA2AddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2AddBindings,
                                                      tileConstraint = XDNA2AddTileConstraint())
XDNA2MulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2MulBindings,
                                                      tileConstraint = XDNA2AddTileConstraint())
XDNA2GeluTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2GeluBindings,
                                                       tileConstraint = XDNA2UnaryTileConstraint())
XDNA2ReluTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2ReluBindings,
                                                       tileConstraint = XDNA2UnaryTileConstraint())
XDNA2SiLUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2SiLUBindings,
                                                       tileConstraint = XDNA2UnaryTileConstraint())
XDNA2TanhTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2TanhBindings,
                                                       tileConstraint = XDNA2UnaryTileConstraint())
XDNA2LayerNormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2LayerNormBindings,
                                                            tileConstraint = XDNA2LayerNormTileConstraint())

# Data movement nodes handled by MemTile with memory tile layout constraints
XDNA2SplitMemTileTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2SplitMemTileBindings,
                                                               tileConstraint = XDNA2MemTileLayoutTileConstraint())
XDNA2ConcatMemTileTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2ConcatMemTileBindings,
                                                                tileConstraint = XDNA2MemTileLayoutTileConstraint())
