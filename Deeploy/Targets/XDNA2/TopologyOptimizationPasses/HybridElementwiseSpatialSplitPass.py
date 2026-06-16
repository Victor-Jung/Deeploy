# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Arity-aware spatial split for XDNA2 elementwise ops.

Subsumes the earlier shim-direct and mem-tile passes into one pass that
decides per-column geometry from the op arity and the per-column row
count R, using the shim's 2-input / 2-output channel budget as the cap.

The shim channel accounting:

* Each shim has 2 MM2S (input direction) and 2 S2MM (output direction)
  channels.
* Mem-tile has 6 input and output channels, data still need to go through the shim but can then be splitted/concatenated through mem-tile. 

Per-column geometry, by arity:

* Binary, R == 1   → 1 direct chunk
* Binary, R >= 2   → 1 mem-tile group fanning to R cores
* Unary,  R == 1   → 1 direct chunk
* Unary,  R == 2   → 2 direct chunks (full shim utilisation)
* Unary,  R == 3   → 1 direct chunk + 1 mem-tile group of 2
* Unary,  R == 4   → 2 mem-tile groups of 2 each

The choice for unary R >= 3 uses two independent mem-tile groups (one
per shim channel pair) instead of one big group; this preserves full
shim bandwidth at the cost of an extra ObjectFifo + link op per column.

For direct groups the group chunk IS the row chunk (1 row per group):
the compute sub-op consumes it through a shim↔core ObjectFifo via the
existing :class:`MLIRObjectFifoPass` path.

Divisibility: ``axis_size`` must be divisible by ``N x R`` (so per-row
chunks have integer length). Non-uniform groups (R=3 → sizes [1, 2])
still satisfy this because each group's chunk is a multiple of the row
chunk size.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import TopologyOptimizationPass
from Deeploy.EngineExtension.OptimizationPasses.EngineAwarePass import engineaware
from Deeploy.Targets.XDNA2.Platform import VECTOR_WIDTH_BF16, XDNA2AIECoreEngine, XDNA2MemTileDataMover, \
    XDNA2MemTileExecutionEngine, XDNA2ShimTileDataMover, next_multiple

_ELEMENTWISE_OPS = frozenset({"Add", "Gelu", "Mul", "Relu", "Silu", "Tanh"})


def _columnLayout(R: int, arity: str) -> List[Tuple[str, List[int]]]:
    """Per-column row groupings constrained by the shim channel budget.

    Returns a list of ``(kind, [local_row_indices])`` tuples; kinds are
    ``"direct"`` (1 row, shim → core) or ``"memtile"`` (G rows, shim →
    mem-tile → cores via a Split/Concat pair). The list spans the full
    R rows of the column with no overlap.
    """
    if arity == "binary":
        if R == 1:
            return [("direct", [0])]
        return [("memtile", list(range(R)))]
    # unary
    if R == 1:
        return [("direct", [0])]
    if R == 2:
        return [("direct", [0]), ("direct", [1])]
    if R == 3:
        return [("direct", [0]), ("memtile", [1, 2])]
    if R == 4:
        return [("memtile", [0, 1]), ("memtile", [2, 3])]
    raise NotImplementedError(
        f"_columnLayout: unary R={R} is outside the supported range (1-4). "
        f"NPU2 has at most 4 AIE rows per column; if you have a different "
        f"target add a case here.")


@engineaware
class XDNA2HybridElementwiseSpatialSplitPass(TopologyOptimizationPass):
    """Unified arity-aware spatial-split pass for XDNA2 elementwise ops.

    Replaces the earlier shim-direct-only and all-mem-tile passes.
    """

    def __init__(self, axis: int = 0) -> None:
        super().__init__()
        self.axis = int(axis)

    # ------------------------------------------------------------------

    def apply(self, graph: gs.Graph) -> Tuple[gs.Graph]:
        coreEngines, memEngines, shims, memMovers = self._partitionEngines()
        assert coreEngines, "No XDNA2AIECoreEngine on the platform."
        assert shims, "No XDNA2ShimTileDataMover on the platform."

        coresByCol: Dict[int, List[XDNA2AIECoreEngine]] = defaultdict(list)
        for e in coreEngines:
            coresByCol[e.col].append(e)
        for col in coresByCol:
            coresByCol[col].sort(key = lambda e: e.row)
        memEngineByCol = {e.col: e for e in memEngines}

        # Active columns: AIE engines present + the optional plumbing
        # needed for mem-tile groups (the per-column shim + mem-tile
        # engines + mem-tile data mover). Columns without that plumbing
        # can still host direct-only layouts (R=1).
        active_cols = sorted(coresByCol.keys())
        active_cols = [c for c in active_cols if c in shims]
        if not active_cols:
            return graph

        R = len(coresByCol[active_cols[0]])
        for c in active_cols:
            assert len(coresByCol[c]) == R, (
                f"Non-uniform AIE row count across columns — this pass "
                f"requires uniform R per column.")

        N = len(active_cols)
        total = N * R

        # Pad graph IO so per-row chunks are a multiple of the vector unit width
        self._padGraphIOForVectorAlignment(graph, total * VECTOR_WIDTH_BF16)

        for node in list(graph.nodes):
            if node.op not in _ELEMENTWISE_OPS:
                continue
            if not self._splittable(node, total):
                continue
            arity = "binary" if len(node.inputs) == 2 else "unary"
            layout = _columnLayout(R, arity)
            # Any mem-tile group needs the column's mem-tile execution
            # engine + mem-tile data mover. Sanity-check they exist.
            if any(kind == "memtile" for kind, _ in layout):
                missing = [c for c in active_cols
                           if c not in memEngineByCol or c not in memMovers]
                assert not missing, (
                    f"Hybrid split for arity={arity} R={R} needs a mem-tile "
                    f"execution engine + mem-tile data mover for columns "
                    f"{missing}, but they're not registered.")
            self._splitElementwise(graph, node, active_cols, coresByCol,
                                   memEngineByCol, memMovers, shims,
                                   layout)

        graph.cleanup().toposort()

        fallback = sorted(coreEngines, key = lambda e: (e.col, e.row))[0].name
        for node in graph.nodes:
            if "engine" not in node.attrs:
                node.attrs["engine"] = fallback

        return graph

    # ------------------------------------------------------------------

    def _partitionEngines(self) -> Tuple[List[XDNA2AIECoreEngine],
                                         List[XDNA2MemTileExecutionEngine],
                                         Dict[int, XDNA2ShimTileDataMover],
                                         Dict[int, XDNA2MemTileDataMover]]:
        engines = getattr(self, "engines", None)
        assert engines is not None, (
            "XDNA2HybridElementwiseSpatialSplitPass.apply called before "
            "EngineColoringDeployer injected the engine list — wrap the "
            "deployer with EngineColoringDeployerWrapper.")

        coreEngines = [e for e in engines if isinstance(e, XDNA2AIECoreEngine)]
        memEngines = [e for e in engines if isinstance(e, XDNA2MemTileExecutionEngine)]

        platformDataMovers = getattr(self, "dataMoverEngines", [])
        shims = {dm.col: dm for dm in platformDataMovers if isinstance(dm, XDNA2ShimTileDataMover)}
        memMovers = {dm.col: dm for dm in platformDataMovers if isinstance(dm, XDNA2MemTileDataMover)}

        return coreEngines, memEngines, shims, memMovers

    def _padGraphIOForVectorAlignment(self, graph: gs.Graph, divisor: int) -> None:
        """Pad graph IO axis sizes to satisfy this pass's divisibility.

        Skipped (and the node will simply not split) when:
          * any input is a constant (would need data extension at compile
            time, deferred to a later iteration),
          * any IO of the node is not a graph input/output (intermediates
            cross other ops and padding them would mismatch sibling
            consumers),
          * axis != 0 (only flat-tail layouts are safe for memset-zero in
            the host today; column-padding for axis>0 is interleaved and
            needs a different host strategy).
        """
        if self.axis != 0:
            # No-op for axis>0; _splittable will reject misaligned nodes as
            # before. Worth lifting in a follow-up if the need arises.
            return

        for node in graph.nodes:
            if node.op not in _ELEMENTWISE_OPS:
                continue
            if len(node.outputs) != 1:
                continue
            out = node.outputs[0]
            if out.shape is None or len(out.shape) <= self.axis:
                continue
            ax = out.shape[self.axis]
            if not isinstance(ax, int):
                continue
            # All IO must share the same axis size for elementwise — this
            # mirrors _splittable's invariant.
            if any(inp.shape is None or len(inp.shape) <= self.axis
                   or inp.shape[self.axis] != ax for inp in node.inputs):
                continue
            # Constants and intermediates: skip; see docstring.
            if any(isinstance(inp, gs.Constant) for inp in node.inputs):
                continue
            if any(inp not in graph.inputs for inp in node.inputs):
                continue
            if out not in graph.outputs:
                continue

            padded_ax = next_multiple(ax, divisor)
            if padded_ax == ax:
                continue

            for var in (*node.inputs, out):
                self._padVariable(var, padded_ax)

    @staticmethod
    def _padVariable(var: gs.Variable, padded_ax: int, axis: int = 0) -> None:
        """Grow ``var.shape[axis]`` to ``padded_ax`` and record the delta.

        ``_paddingElems`` is the count of TAIL elements added (flat
        memory), suitable for ``memset(buf + logical, 0, ...)`` on the
        host. With axis=0 and row-major layout the tail interpretation is
        exact; this helper asserts that invariant.
        """
        assert axis == 0, "padding tail layout only valid for axis=0 today"
        new_shape = list(var.shape)
        old_ax = new_shape[axis]
        if padded_ax <= old_ax:
            return
        new_shape[axis] = padded_ax
        # Use the max of any pre-existing padding contribution (another
        # pass may have already requested more). Padding combines via max
        # of the *axis size*, not of the per-tensor tail-element count.
        existing = getattr(var, "_paddingElems", 0) or 0
        other_dims = int(np.prod(new_shape[1:])) if len(new_shape) > 1 else 1
        added = (padded_ax - old_ax) * other_dims
        var.shape = tuple(new_shape)
        var._paddingElems = max(int(existing), added)

    def _splittable(self, node: gs.Node, total_chunks: int) -> bool:
        if len(node.outputs) != 1:
            return False
        out = node.outputs[0]
        if out.shape is None or len(out.shape) <= self.axis:
            return False
        ax = out.shape[self.axis]
        if not isinstance(ax, int) or ax % total_chunks != 0:
            return False
        for inp in node.inputs:
            if inp.shape is None or len(inp.shape) <= self.axis or inp.shape[self.axis] != ax:
                return False
        return True

    def _chunkShape(self, shape, axis_chunks: int):
        """Replace `shape[axis]` with `shape[axis] // axis_chunks`."""
        chunked = list(shape)
        chunked[self.axis] = chunked[self.axis] // axis_chunks
        return tuple(chunked)

    def _replaceAxis(self, shape, new_axis_size: int):
        chunked = list(shape)
        chunked[self.axis] = new_axis_size
        return tuple(chunked)

    # ------------------------------------------------------------------

    def _splitElementwise(self, graph: gs.Graph, node: gs.Node,
                          active_cols: List[int],
                          coresByCol: Dict[int, List[XDNA2AIECoreEngine]],
                          memEngineByCol: Dict[int, XDNA2MemTileExecutionEngine],
                          memMovers: Dict[int, XDNA2MemTileDataMover],
                          shims: Dict[int, XDNA2ShimTileDataMover],
                          layout: List[Tuple[str, List[int]]]) -> None:
        N = len(active_cols)
        R = sum(len(rows) for _, rows in layout)
        op = node.op
        baseName = node.name or f"{op}_{id(node):x}"

        original_inputs = list(node.inputs)
        original_output = node.outputs[0]

        # Per-row chunk shape — the unit every core consumes/produces.
        # Per-group chunks (what the shim transfers) are a contiguous
        # block of contiguous rows on the split axis.
        row_in_shapes = [self._chunkShape(inp.shape, N * R) for inp in original_inputs]
        row_out_shape = self._chunkShape(original_output.shape, N * R)
        row_in_elems = [int(np.prod(s)) for s in row_in_shapes]
        row_out_elems = int(np.prod(row_out_shape))

        # Detach original op.
        node.inputs.clear()
        node.outputs.clear()
        graph.nodes.remove(node)

        # Tracks created sub-op outputs per (input/output, c_idx, local_row)
        # so we can wire each sub-op's inputs and outputs correctly when
        # the per-input loop is finished.
        # in_row_chunk_map[input_idx][(c_idx, local_row)] = chunk variable
        in_row_chunk_map: List[Dict[Tuple[int, int], gs.Variable]] = [
            {} for _ in original_inputs
        ]
        # out_row_chunk_map[(c_idx, local_row)] = chunk variable
        out_row_chunk_map: Dict[Tuple[int, int], gs.Variable] = {}

        # Collect the (col-chunk-style) graph IO that will replace the
        # original input/output tensors. Order: per active column, then
        # per group within that column. Each group contributes ONE chunk
        # to graph IO (the shim's view).
        new_in_chunks: List[List[gs.Variable]] = [[] for _ in original_inputs]
        new_out_chunks: List[gs.Variable] = []

        for c_idx, c in enumerate(active_cols):
            col_row_offset = 0  # cumulative rows consumed by previous groups in this col
            for g_idx, (kind, local_rows) in enumerate(layout):
                G = len(local_rows)
                # Per-group chunk: a slab of G contiguous rows on the
                # split axis. The shim transfers this in one DMA cycle.
                group_in_shapes = [self._replaceAxis(inp.shape, G * (inp.shape[self.axis] // (N * R)))
                                   for inp in original_inputs]
                group_out_shape = self._replaceAxis(original_output.shape,
                                                    G * (original_output.shape[self.axis] // (N * R)))
                group_in_elems = [int(np.prod(s)) for s in group_in_shapes]
                group_out_elems = int(np.prod(group_out_shape))

                # Offsets relative to the original logical tensor.
                col_chunk_elems_in = [(inp.shape[self.axis] // N) * int(np.prod(inp.shape[:self.axis] + inp.shape[self.axis + 1:]))
                                      for inp in original_inputs]
                col_chunk_elems_out = (original_output.shape[self.axis] // N) * int(np.prod(
                    original_output.shape[:self.axis] + original_output.shape[self.axis + 1:]))
                group_offset_in_col_in = [col_row_offset * row_in_elems[i] for i in range(len(original_inputs))]
                group_offset_in_col_out = col_row_offset * row_out_elems

                # ---- Per-input group chunk + (optional) Split ----
                for inp_idx, inp in enumerate(original_inputs):
                    gc = gs.Variable(
                        name = f"{inp.name}_c{c}_g{g_idx}",
                        dtype = inp.dtype,
                        shape = group_in_shapes[inp_idx],
                    )
                    gc._logicalParent = inp.name
                    gc._chunkOffset = c_idx * col_chunk_elems_in[inp_idx] + group_offset_in_col_in[inp_idx]
                    gc._dataMoverEngine = shims[c].name
                    new_in_chunks[inp_idx].append(gc)

                    if kind == "direct":
                        # Group is 1 row — the group chunk IS the row chunk.
                        # The sub-op will consume it directly through a
                        # shim↔core FIFO.
                        assert G == 1
                        in_row_chunk_map[inp_idx][(c_idx, local_rows[0])] = gc
                    else:
                        # Mem-tile group — Split fans the group chunk to
                        # G row chunks via the column's mem-tile DMA.
                        row_chunks = []
                        for j, lr in enumerate(local_rows):
                            rc = gs.Variable(
                                name = f"{inp.name}_c{c}_g{g_idx}_r{lr}",
                                dtype = inp.dtype,
                                shape = row_in_shapes[inp_idx],
                            )
                            rc._logicalParent = gc.name
                            rc._chunkOffset = j * row_in_elems[inp_idx]
                            rc._dataMoverEngine = memMovers[c].name
                            row_chunks.append(rc)
                            in_row_chunk_map[inp_idx][(c_idx, lr)] = rc
                        split = gs.Node(
                            op = "Split",
                            name = f"{baseName}_c{c}_g{g_idx}_split_in{inp_idx}",
                            inputs = [gc],
                            outputs = row_chunks,
                            attrs = {"axis": self.axis, "engine": memEngineByCol[c].name},
                        )
                        graph.nodes.append(split)

                # ---- Output group chunk + (optional) Concat ----
                ogc = gs.Variable(
                    name = f"{original_output.name}_c{c}_g{g_idx}",
                    dtype = original_output.dtype,
                    shape = group_out_shape,
                )
                ogc._logicalParent = original_output.name
                ogc._chunkOffset = c_idx * col_chunk_elems_out + group_offset_in_col_out
                ogc._dataMoverEngine = shims[c].name
                new_out_chunks.append(ogc)

                if kind == "direct":
                    assert G == 1
                    out_row_chunk_map[(c_idx, local_rows[0])] = ogc
                else:
                    row_chunks_out = []
                    for j, lr in enumerate(local_rows):
                        orc = gs.Variable(
                            name = f"{original_output.name}_c{c}_g{g_idx}_r{lr}",
                            dtype = original_output.dtype,
                            shape = row_out_shape,
                        )
                        orc._logicalParent = ogc.name
                        orc._chunkOffset = j * row_out_elems
                        orc._dataMoverEngine = memMovers[c].name
                        row_chunks_out.append(orc)
                        out_row_chunk_map[(c_idx, lr)] = orc
                    concat = gs.Node(
                        op = "Concat",
                        name = f"{baseName}_c{c}_g{g_idx}_concat_out",
                        inputs = row_chunks_out,
                        outputs = [ogc],
                        attrs = {"axis": self.axis, "engine": memEngineByCol[c].name},
                    )
                    graph.nodes.append(concat)

                col_row_offset += G

        # ---- Sub-ops per (col, local_row) ----
        for c_idx, c in enumerate(active_cols):
            for local_row in range(R):
                core_engine = coresByCol[c][local_row].name
                sub_inputs = [in_row_chunk_map[i][(c_idx, local_row)]
                              for i in range(len(original_inputs))]
                sub_output = out_row_chunk_map[(c_idx, local_row)]
                sub = gs.Node(
                    op = op,
                    name = f"{baseName}_c{c}_r{local_row}",
                    inputs = sub_inputs,
                    outputs = [sub_output],
                    attrs = {**{k: v for k, v in node.attrs.items() if k != "engine"},
                             "engine": core_engine},
                )
                graph.nodes.append(sub)

        # ---- Replace original graph IO with the new group chunks ----
        for inp_idx, inp in enumerate(original_inputs):
            self._replaceInGraphInputs(graph, inp, new_in_chunks[inp_idx])
        self._replaceInGraphOutputs(graph, original_output, new_out_chunks)

    # ------------------------------------------------------------------

    @staticmethod
    def _replaceInGraphInputs(graph: gs.Graph, original: gs.Variable,
                              replacements: List[gs.Variable]) -> None:
        if original not in graph.inputs:
            raise NotImplementedError(
                "Hybrid spatial split currently only handles inputs that are graph inputs. "
                f"Tensor '{original.name}' is an intermediate.")
        idx = graph.inputs.index(original)
        graph.inputs[idx:idx + 1] = replacements

    @staticmethod
    def _replaceInGraphOutputs(graph: gs.Graph, original: gs.Variable,
                               replacements: List[gs.Variable]) -> None:
        if original not in graph.outputs:
            raise NotImplementedError(
                "Hybrid spatial split currently only handles outputs that are graph outputs. "
                f"Tensor '{original.name}' is an intermediate.")
        idx = graph.outputs.index(original)
        graph.outputs[idx:idx + 1] = replacements
