# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Spatial mapping pass for XDNA2.

Split eligible elementwise nodes (e.g. ``Add``, ``Silu``) across compute
cores and annotate the resulting per-chunk tensors with their data movers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import TopologyOptimizationPass
from Deeploy.EngineExtension.OptimizationPasses.EngineAwarePass import engineaware
from Deeploy.Targets.XDNA2.Platform import XDNA2AIECoreDataMover, XDNA2AIECoreEngine, XDNA2ShimTileDataMover


_ELEMENTWISE_OPS = frozenset({"Add", "Gelu", "Mul", "Relu", "Silu", "Tanh"})


@engineaware
class XDNA2ElementwiseSpatialSplitPass(TopologyOptimizationPass):
    """Annotate graph IO with data movers; split eligible elementwise nodes across cores.

    Parameters
    ----------
    axis : int
        Tensor axis along which to split splittable ops. Default 0.
    """

    def __init__(self, axis: int = 0) -> None:
        super().__init__()
        self.axis = int(axis)

    def apply(self, graph: gs.Graph) -> Tuple[gs.Graph]:
        coreEngines, shims, _coreDataMovers = self._partitionEngines()
        assert coreEngines, "No XDNA2AIECoreEngine on the platform — cannot map compute nodes."
        assert shims, "No XDNA2ShimTileDataMover on the platform — cannot move data to/from L3."

        # Try to spatially split eligible compute nodes. The chunks introduced
        # here are annotated with their per-column shim; original tensors that
        # get replaced by chunks lose their default annotation along with
        # them. Tensors not touched by a split keep the default mover that
        # XDNA2DefaultDataMoverPass installed.
        if len(coreEngines) >= 2:
            for node in list(graph.nodes):
                if node.op not in _ELEMENTWISE_OPS:
                    continue
                if not self._splittable(node, len(coreEngines)):
                    continue
                self._splitElementwise(graph, node, coreEngines, shims)
            graph.cleanup().toposort()

        # Color compute nodes that aren't already colored. The split pass
        # handles its own sub-ops; this picks up the LayerNorm / un-split
        # elementwise case.
        for node in graph.nodes:
            if "engine" not in node.attrs:
                node.attrs["engine"] = coreEngines[0].name

        return graph

    def _partitionEngines(self) -> Tuple[List[XDNA2AIECoreEngine], List[XDNA2ShimTileDataMover],
                                         Dict[Tuple[int, int], XDNA2AIECoreDataMover]]:
        engines = getattr(self, "engines", None)
        assert engines is not None, (
            "XDNA2ElementwiseSpatialSplitPass.apply called before EngineColoringDeployer "
            "injected the engine list — wrap the deployer with "
            "EngineColoringDeployerWrapper.")

        coreEngines = sorted([e for e in engines if isinstance(e, XDNA2AIECoreEngine)],
                             key = lambda e: (e.col, e.row))
        # Data movers come in via the platform's separate dataMoverEngines list.
        # The engineaware mixin only injects the compute engine list, so we
        # recover data movers from the platform attribute (set on the bound
        # pass instance by the deployer).
        platformDataMovers = getattr(self, "dataMoverEngines", [])
        shims = sorted([dm for dm in platformDataMovers if isinstance(dm, XDNA2ShimTileDataMover)],
                       key = lambda dm: dm.col)
        coreDataMovers = {(dm.col, dm.row): dm
                          for dm in platformDataMovers
                          if isinstance(dm, XDNA2AIECoreDataMover)}
        return coreEngines, shims, coreDataMovers

    def _splittable(self, node: gs.Node, num_cores: int) -> bool:
        if len(node.outputs) != 1:
            return False
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

    def _chunkShape(self, shape: Tuple[int, ...], num_chunks: int) -> Tuple[int, ...]:
        chunked = list(shape)
        chunked[self.axis] = chunked[self.axis] // num_chunks
        return tuple(chunked)

    def _splitElementwise(self, graph: gs.Graph, node: gs.Node, coreEngines: List[XDNA2AIECoreEngine],
                          shims: List[XDNA2ShimTileDataMover]) -> None:
        """Replace ``node`` (an elementwise op) with ``len(coreEngines)`` sub-ops.

        Each sub-op lives on its own AIE compute tile. Inputs/outputs are
        split into per-chunk graph IO tensors so the IR stays
        single-producer / single-consumer (the tiler rejects multi-producer).
        """
        num_cores = len(coreEngines)
        op = node.op
        baseName = node.name or f"{op}_{id(node):x}"

        # Lookup table: column → its shim data mover.
        shim_by_col = {dm.col: dm for dm in shims}

        original_inputs = list(node.inputs)
        original_output = node.outputs[0]
        out_chunk_shape = self._chunkShape(original_output.shape, num_cores)
        in_chunk_shapes = [self._chunkShape(inp.shape, num_cores) for inp in original_inputs]

        # Detach the node from the graph; replacement nodes go in place.
        node.inputs.clear()
        node.outputs.clear()
        graph.nodes.remove(node)

        # Per-input chunk element count (chunks for one logical input all
        # have the same shape, so one count per input is enough).
        in_chunk_elems = [int(np.prod(s)) for s in in_chunk_shapes]
        out_chunk_elems = int(np.prod(out_chunk_shape))

        # For each input that was a graph input, split it into N chunks.
        # Each chunk REPLACES the original in graph.inputs.
        per_input_chunks: List[List[gs.Variable]] = []
        for inp_idx, inp in enumerate(original_inputs):
            chunks: List[gs.Variable] = []
            for i in range(num_cores):
                chunk = gs.Variable(
                    name = f"{inp.name}_c{i}",
                    dtype = inp.dtype,
                    shape = in_chunk_shapes[inp_idx],
                )
                chunk._dataMoverEngine = shim_by_col[coreEngines[i].col].name
                chunk._logicalParent = inp.name
                chunk._chunkOffset = i * in_chunk_elems[inp_idx]
                chunks.append(chunk)
            per_input_chunks.append(chunks)
            self._replaceInGraphInputs(graph, inp, chunks)

        # Same for the output.
        out_chunks: List[gs.Variable] = []
        for i in range(num_cores):
            chunk = gs.Variable(
                name = f"{original_output.name}_c{i}",
                dtype = original_output.dtype,
                shape = out_chunk_shape,
            )
            chunk._dataMoverEngine = shim_by_col[coreEngines[i].col].name
            chunk._logicalParent = original_output.name
            chunk._chunkOffset = i * out_chunk_elems
            out_chunks.append(chunk)
        self._replaceInGraphOutputs(graph, original_output, out_chunks)

        # Emit the N sub-ops, colored to their respective AIE cores.
        for i in range(num_cores):
            sub = gs.Node(
                op = op,
                name = f"{baseName}_core{i}",
                inputs = [per_input_chunks[inp_idx][i] for inp_idx in range(len(original_inputs))],
                outputs = [out_chunks[i]],
                attrs = {**{k: v for k, v in node.attrs.items() if k != "engine"}, "engine": coreEngines[i].name},
            )
            graph.nodes.append(sub)

    @staticmethod
    def _replaceInGraphInputs(graph: gs.Graph, original: gs.Variable, replacements: List[gs.Variable]) -> None:
        if original not in graph.inputs:
            raise NotImplementedError(
                "XDNA2ElementwiseSpatialSplitPass currently only splits inputs that are graph inputs. "
                f"Tensor '{original.name}' is an intermediate.")
        idx = graph.inputs.index(original)
        graph.inputs[idx:idx + 1] = replacements

    @staticmethod
    def _replaceInGraphOutputs(graph: gs.Graph, original: gs.Variable, replacements: List[gs.Variable]) -> None:
        if original not in graph.outputs:
            raise NotImplementedError(
                "XDNA2ElementwiseSpatialSplitPass currently only splits outputs that are graph outputs. "
                f"Tensor '{original.name}' is an intermediate.")
        idx = graph.outputs.index(original)
        graph.outputs[idx:idx + 1] = replacements
