# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Spatial mapping pass for XDNA2: wrap every memory transition in a shim node.

Two responsibilities:

1. **Always wrap graph IO with shim nodes.** Every input edge entering a
   compute node from L3 (graph inputs) becomes ``ShimRead(L3) -> L1_chunk``,
   and every output edge from a compute node to L3 (graph outputs) becomes
   ``ShimWrite(L1_chunk) -> L3``. This holds even with no spatial split —
   single-core mode also gets shim nodes. The principle is "every
   memory-level transition is a node".

2. **Spatially split eligible compute nodes across AIE cores.** When the
   platform exposes N >= 2 ``XDNA2AIECoreEngine`` instances, an eligible
   ``Add`` is replaced by N sub-``Add``s (one per core). Non-Add ops are
   wrapped without splitting (chunk count = 1, mapped to the first core).

Each shim node is colored to a ``XDNA2ShimEngine`` in the column of its
associated compute core; each compute (sub-)node is colored to its
``XDNA2AIECoreEngine``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import TopologyOptimizationPass
from Deeploy.EngineExtension.OptimizationPasses.EngineAwarePass import engineaware
from Deeploy.Targets.XDNA2.Platform import XDNA2AIECoreEngine, XDNA2ShimEngine

# Compute op kinds we know how to wrap. Ops not in this set are passed
# through unchanged — we'll need to extend this as more kernels land.
_COMPUTE_OPS = frozenset({"Add", "Silu", "LayerNormalization"})

_SPLITTABLE_OPS = frozenset({"Add"})


@engineaware
class XDNA2SpatialSplitPass(TopologyOptimizationPass):
    """Wrap compute ops in shim nodes; split eligible ones across AIE cores.

    Parameters
    ----------
    axis : int
        Tensor axis along which to split. Default 0.
    """

    def __init__(self, axis: int = 0) -> None:
        super().__init__()
        self.axis = int(axis)

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    def apply(self, graph: gs.Graph) -> Tuple[gs.Graph]:
        coreEngines, shimByCol = self._partitionEngines()
        assert coreEngines, "No XDNA2AIECoreEngine on the platform — cannot map compute nodes."

        for node in list(graph.nodes):
            if node.op not in _COMPUTE_OPS:
                continue
            num_chunks = len(coreEngines) if node.op in _SPLITTABLE_OPS and self._splittable(node, len(coreEngines)) \
                else 1
            self._wrapAndSplit(graph, node, coreEngines, shimByCol, num_chunks)

        graph.cleanup().toposort()
        return graph

    # ------------------------------------------------------------------
    # engine bookkeeping
    # ------------------------------------------------------------------

    def _partitionEngines(self) -> Tuple[List[XDNA2AIECoreEngine], Dict[int, XDNA2ShimEngine]]:
        engines = getattr(self, "engines", None)
        assert engines is not None, (
            "XDNA2SpatialSplitPass.apply called before EngineColoringDeployer "
            "injected the engine list — wrap the deployer with "
            "EngineColoringDeployerWrapper.")
        coreEngines = [e for e in engines if isinstance(e, XDNA2AIECoreEngine)]
        shimByCol = {e.col: e for e in engines if isinstance(e, XDNA2ShimEngine)}
        for core in coreEngines:
            assert core.col in shimByCol, (
                f"No XDNA2ShimEngine for column {core.col}; the platform must expose "
                f"one shim engine per AIE core column.")
        return coreEngines, shimByCol

    # ------------------------------------------------------------------
    # eligibility
    # ------------------------------------------------------------------

    def _splittable(self, node: gs.Node, num_cores: int) -> bool:
        """Whether this Add can be evenly split into ``num_cores`` chunks along ``axis``."""
        out = node.outputs[0]
        if out.shape is None or len(out.shape) <= self.axis:
            return False
        ax_dim = out.shape[self.axis]
        if not isinstance(ax_dim, int) or ax_dim % num_cores != 0:
            return False
        for inp in node.inputs:
            if inp.shape is None or len(inp.shape) <= self.axis or inp.shape[self.axis] != ax_dim:
                return False
        return True

    # ------------------------------------------------------------------
    # rewrite
    # ------------------------------------------------------------------

    def _chunkShape(self, shape: Tuple[int, ...], num_chunks: int) -> Tuple[int, ...]:
        if num_chunks == 1:
            return tuple(shape)
        chunked = list(shape)
        chunked[self.axis] = chunked[self.axis] // num_chunks
        return tuple(chunked)

    def _wrapAndSplit(self, graph: gs.Graph, node: gs.Node, coreEngines: List[XDNA2AIECoreEngine],
                      shimByCol: Dict[int, XDNA2ShimEngine], num_chunks: int) -> None:
        """Replace ``node`` with ``num_chunks`` shim-wrapped copies."""
        baseName = node.name or f"{node.op}_{id(node):x}"
        inputs = list(node.inputs)
        output = node.outputs[0]
        out_chunk_shape = self._chunkShape(output.shape, num_chunks)
        in_chunk_shapes = [self._chunkShape(inp.shape, num_chunks) for inp in inputs]
        # Element count per chunk (used as DMA length and offset multiplier).
        out_chunk_elems = 1
        for d in out_chunk_shape:
            out_chunk_elems *= int(d)
        in_chunk_elems = []
        for shp in in_chunk_shapes:
            n = 1
            for d in shp:
                n *= int(d)
            in_chunk_elems.append(n)

        new_nodes: List[gs.Node] = []
        shim_write_outs: List[gs.Variable] = []

        for i in range(num_chunks):
            core = coreEngines[i] if num_chunks > 1 else coreEngines[0]
            shim = shimByCol[core.col]

            # Per-input ShimRead — one per (input, chunk) combination.
            in_chunks = []
            for inp_idx, (inp, shp, elems) in enumerate(zip(inputs, in_chunk_shapes, in_chunk_elems)):
                in_chunk = gs.Variable(name = f"{baseName}_in{inp_idx}_c{i}", dtype = inp.dtype, shape = shp)
                shim_read = gs.Node(
                    op = "ShimRead",
                    name = f"{baseName}_ShimRead_in{inp_idx}_c{i}",
                    inputs = [inp],
                    outputs = [in_chunk],
                    attrs = {
                        "engine": shim.name,
                        "axis": self.axis,
                        "offset": i * elems,
                        "length": elems,
                    },
                )
                in_chunks.append(in_chunk)
                new_nodes.append(shim_read)

            # The compute (sub-)node, colored to its AIE core.
            out_chunk = gs.Variable(name = f"{baseName}_out_c{i}", dtype = output.dtype, shape = out_chunk_shape)
            sub_node = gs.Node(
                op = node.op,
                name = f"{baseName}_core{i}" if num_chunks > 1 else baseName,
                inputs = in_chunks,
                outputs = [out_chunk],
                # Preserve any non-engine attrs from the original node (e.g. epsilon for LayerNorm).
                attrs = {**{k: v for k, v in node.attrs.items() if k != "engine"}, "engine": core.name},
            )
            new_nodes.append(sub_node)

            # ShimWrite drains the L1 chunk back to L3.
            #  * num_chunks == 1: writes the graph output tensor directly.
            #  * num_chunks  > 1: writes a per-chunk intermediate; a Concat
            #    marker (added below) reconstructs the graph output. This
            #    avoids the multi-producer assertion in the tiler.
            if num_chunks == 1:
                shim_write_dst = output
            else:
                shim_write_dst = gs.Variable(
                    name = f"{baseName}_out_l3_c{i}",
                    dtype = output.dtype,
                    shape = out_chunk_shape,
                )
            shim_write = gs.Node(
                op = "ShimWrite",
                name = f"{baseName}_ShimWrite_c{i}",
                inputs = [out_chunk],
                outputs = [shim_write_dst],
                attrs = {
                    "engine": shim.name,
                    "axis": self.axis,
                    "offset": i * out_chunk_elems,
                    "length": out_chunk_elems,
                },
            )
            new_nodes.append(shim_write)
            shim_write_outs.append(shim_write_dst)

        # Concat marker — only for num_chunks > 1. Colored to the first
        # shim engine; emits no MLIR, exists purely to keep the graph
        # single-producer.
        if num_chunks > 1:
            concat_shim = shimByCol[coreEngines[0].col]
            concat = gs.Node(
                op = "Concat",
                name = f"{baseName}_Concat",
                inputs = shim_write_outs,
                outputs = [output],
                attrs = {"engine": concat_shim.name, "axis": self.axis},
            )
            new_nodes.append(concat)

        node.inputs.clear()
        node.outputs.clear()
        graph.nodes.remove(node)
        graph.nodes.extend(new_nodes)
