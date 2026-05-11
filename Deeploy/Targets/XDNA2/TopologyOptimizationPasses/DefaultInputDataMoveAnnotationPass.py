# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Default data-mover annotation for graph inputs and outputs.

Seeds every graph input and output ``gs.Variable`` with a default
:class:`XDNA2ShimTileDataMover` (the lowest-column shim) when no other
pass has assigned one yet.
"""

from __future__ import annotations

from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import TopologyOptimizationPass
from Deeploy.EngineExtension.OptimizationPasses.EngineAwarePass import engineaware
from Deeploy.Targets.XDNA2.Platform import XDNA2ShimTileDataMover


@engineaware
class XDNA2DefaultIODataMoveAnnotationPass(TopologyOptimizationPass):
    """Seed graph inputs and outputs with a default L3 shim mover."""

    def apply(self, graph: gs.Graph) -> Tuple[gs.Graph]:
        shims = sorted(
            [dm for dm in getattr(self, "dataMoverEngines", []) if isinstance(dm, XDNA2ShimTileDataMover)],
            key = lambda dm: dm.col,
        )
        assert shims, ("No XDNA2ShimTileDataMover on the platform — at least one is required "
                       "for the default data-mover annotation.")
        defaultName = shims[0].name

        for tensor in list(graph.inputs) + list(graph.outputs):
            if getattr(tensor, "_dataMoverEngine", None) is None:
                tensor._dataMoverEngine = defaultName

        return graph
