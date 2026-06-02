# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Base classes for MLIR-emitting node templates and code transformations.

This module provides:

* :class:`MLIRNodeTemplate` — a :class:`NodeTemplate` subclass whose
  ``emit()`` method populates an ``mlir.ir.Module`` instead of rendering C.
* :class:`MLIRExecutionBlock` — MLIR-specific execution state replacing the
  C-oriented :class:`ExecutionBlock` (code-snippet deque) with MLIR builder
  state (tile references, ObjectFifo handles, tiling parameters).
* :class:`MLIRCodeTransformationPass` — base class for MLIR code
  transformation passes that operate on an :class:`MLIRExecutionBlock`.
* :class:`MLIRCodeTransformation` — two-phase pass container
  (``devicePasses`` + ``runtimeSequencePasses``) that the deployer
  orchestrates inside ``@aie_d.device`` and ``@aiex_d.runtime_sequence``
  regions respectively.

All classes are intentionally dialect-agnostic so that future MLIR-based
backends (NVGPU, Linalg, …) can reuse them.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation


class MLIRExecutionBlock:
    """MLIR-specific execution state for a single operator.

    Replaces the C-oriented :class:`ExecutionBlock` (which holds a deque of
    :class:`CodeSnippet` objects) with fields that carry MLIR builder state
    through the code-transformation pipeline.

    Passes populate fields progressively:

    1. The deployer sets ``computeTile``, ``shimTile``,
       ``operatorRepresentation``, and ``patternMemoryConstraint``.
    2. A device-phase pass (e.g. ``MLIRObjectFifoPass``) fills
       ``fifoMap``, ``fifoTypes``, ``tileSize``, ``numTiles``,
       ``kernelFuncName``, and ``kernelObjFile``.
    3. The deployer sets ``runtimeSequenceArgs`` before the runtime-
       sequence phase.
    4. A runtime-sequence pass (e.g. ``MLIRRuntimeSequencePass``) reads
       all of the above to emit DMA configuration.
    """

    def __init__(self, computeTile: Any = None, shimTile: Any = None) -> None:
        # MLIR tile references (set by deployer)
        self.computeTile: Any = computeTile
        self.shimTile: Any = shimTile

        # Operator metadata (set by deployer from parser)
        self.operatorRepresentation: OperatorRepresentation = {}

        # Tiling constraint from midend solver (may be None)
        self.patternMemoryConstraint: Any = None

        # Populated by device-phase passes (e.g. MLIRObjectFifoPass)
        self.fifoMap: Dict[str, str] = {}  # tensor name → FIFO name
        self.fifoTypes: Dict[str, Any] = {}  # tensor name → MemRefType
        self.tileShape: Tuple[int, ...] = ()  # N-D tile shape from solver
        self.tileSize: int = 0  # np.prod(tileShape) — flat element count
        self.numTiles: int = 0
        self.numElements: int = 0
        self.kernelFuncName: Optional[str] = None
        self.kernelObjFile: Optional[str] = None

        # The MLIRNodeTemplate for this node (set by deployer, called by
        # MLIRComputeCorePass to emit the kernel call inside the core block)
        self.template: Optional[Any] = None

        # Set by deployer before runtime-sequence phase
        self.runtimeSequenceArgs: List[Any] = []

        # Input / output tensor name lists (set by deployer from parser)
        self.inputNames: List[str] = []
        self.outputNames: List[str] = []

        # Trace support (populated by device-phase trace passes, read by
        # runtime-sequence trace pass)
        self.traceConfigs: List[str] = []
        self.traceBufferSize: int = 0
        self.traceShimCol: int = 0

        # Shared FIFO registry across all MLIRExecutionBlocks of one
        # @aie_d.device(...) block, used by mem-tile-engine spatial
        # split. The deployer sets this to a single dict instance shared
        # by every block in the device. Distribute/Join passes (running
        # on Split/Concat memtile-engine nodes) populate it with small
        # mem-tile↔core ObjectFifo names keyed by
        # ``(logical_parent_tensor_name, col, row)``. The compute-node
        # MLIRObjectFifoPass consults it and reuses the registered FIFO
        # name instead of creating a fresh shim↔core FIFO when a port
        # has a hit.
        self.fifoRegistry: Dict[Tuple[str, int, int], str] = {}


class MLIRCodeTransformationPass:
    """Base class for passes that transform an :class:`MLIRExecutionBlock`.

    Subclasses override :meth:`apply` to read / mutate the block's fields
    and optionally emit MLIR operations into the current insertion point.
    """

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        return ctxt, mlirBlock


class MLIRCodeTransformation:
    """Two-phase pass container for MLIR code transformations.

    *devicePasses* run inside an ``@aie_d.device(...)`` region (ObjectFifo
    creation, external-kernel declarations, …).

    *runtimeSequencePasses* run inside an ``@aiex_d.runtime_sequence``
    block (DMA configuration, token await, …).

    The deployer calls :meth:`applyDevicePasses` and
    :meth:`applyRuntimeSequencePasses` at the appropriate points.
    """

    def __init__(self,
                 devicePasses: Optional[List[MLIRCodeTransformationPass]] = None,
                 runtimeSequencePasses: Optional[List[MLIRCodeTransformationPass]] = None) -> None:
        self.devicePasses: List[MLIRCodeTransformationPass] = devicePasses or []
        self.runtimeSequencePasses: List[MLIRCodeTransformationPass] = runtimeSequencePasses or []

    def applyDevicePasses(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
                          name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        for _pass in self.devicePasses:
            ctxt, mlirBlock = _pass.apply(ctxt, mlirBlock, name)
        return ctxt, mlirBlock

    def applyRuntimeSequencePasses(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
                                   name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        for _pass in self.runtimeSequencePasses:
            ctxt, mlirBlock = _pass.apply(ctxt, mlirBlock, name)
        return ctxt, mlirBlock


class GraphAwareNetworkContext(NetworkContext):
    """NetworkContext subclass that exposes node-level navigation helpers such that ExecutionBlocks can have some vision on their predecessors."""

    def populateNameToProducer(self, layerBinding) -> None:
        """One-time setup: stash the layer binding and build the
        name → producer_node_name reverse index. Must be called before
        :meth:`lookupOpRepr` / :meth:`lookupConsumerOpRepr` /
        :meth:`lookupProducerOpRepr`.
        """
        self._layerBinding = layerBinding
        # Skips constants / graph inputs (no producer node).
        self._nameToProducer: Dict[str, str] = {}
        for nodeName, layer in layerBinding.items():
            for out_var in layer.node.outputs:
                self._nameToProducer[out_var.name] = nodeName

    def lookupOpRepr(self, node_name: str):
        """Return the parsed ``operatorRepresentation`` for ``node_name``."""
        binding = getattr(self, "_layerBinding", None)
        assert binding is not None, (
            "GraphAwareNetworkContext.lookupOpRepr called before "
            "populateNameToProducer. Call it once after layer binding is built.")
        if node_name not in binding:
            raise KeyError(f"lookupOpRepr: no layer binding for node '{node_name}'.")
        return binding[node_name].mapper.parser.operatorRepresentation

    def lookupConsumerOpRepr(self, buf_name: str):
        """Return the ``operatorRepresentation`` of the unique consumer of ``buf_name``.

        Asserts ``len(buf._users) == 1``; multi-consumer buffers need
        explicit per-consumer handling at the call site.
        """
        users = self.lookup(buf_name)._users
        assert len(users) == 1, (
            f"lookupConsumerOpRepr: buffer '{buf_name}' has {len(users)} consumers; "
            f"expected exactly 1. Multi-consumer chunks need explicit per-consumer handling.")
        return self.lookupOpRepr(users[0])

    def lookupProducerOpRepr(self, buf_name: str):
        """Return the ``operatorRepresentation`` of the producer of ``buf_name``."""
        producer = getattr(self, "_nameToProducer", {}).get(buf_name)
        if producer is None:
            raise KeyError(
                f"lookupProducerOpRepr: '{buf_name}' has no producer in layerBinding "
                f"(graph input or constant?).")
        return self.lookupOpRepr(producer)


class MLIRNodeTemplate(NodeTemplate):
    """NodeTemplate subclass that emits MLIR instead of C code.

    Subclasses must override :meth:`emit` to add dialect operations to an
    ``mlir.ir.Module`` (or region / insertion point provided via *kwargs*).

    ``generate()`` is overridden as a convenience that constructs a
    standalone module, calls :meth:`emit`, and returns the MLIR text.
    The base-class ``alignToContext`` / ``hoistTransientBuffers`` hooks are
    retained and work unchanged.
    """

    # Subclasses MUST set these class attributes:
    KERNEL_FN: str = ""  # External kernel function name (e.g. "silu_bf16")
    KERNEL_OBJ: str = ""  # Kernel object file (e.g. "silu.o")
    INPUT_KEYS: List[str] = []  # Keys in operatorRepresentation for input tensors
    OUTPUT_KEYS: List[str] = []  # Keys in operatorRepresentation for output tensors

    def __init__(self):
        # Empty Mako template — no C code is generated.
        super().__init__("")

    def kernelArgTypes(self, tileTy: Any) -> List[Any]:
        """Return the MLIR argument types for the external kernel declaration.

        Default: one memref per input + output tensor, plus a trailing i32 size.
        Override for non-standard kernel signatures.
        """
        import aie.ir as ir
        i32 = ir.IntegerType.get_signless(32)
        return [tileTy] * (len(self.INPUT_KEYS) + len(self.OUTPUT_KEYS)) + [i32]

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------

    @abstractmethod
    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        """Populate an MLIR module with the operations for this node.

        The caller (typically the deployer) sets up an ``mlir.ir.Module``
        with the appropriate device wrapper and passes dialect-specific
        context through *kwargs* (e.g. insertion point, tile references,
        ObjectFifo handles).

        Parameters
        ----------
        operatorRepresentation : OperatorRepresentation
            The parser's node representation (buffer names, sizes, types …).
        **kwargs
            Dialect-specific context provided by the deployer.
        """
        ...


    def generate(self, operatorRepresentation = {}, **kwargs) -> str:
        """Generate an MLIR string for this node.

        This default implementation is a thin wrapper: it delegates to
        :meth:`emit`.  Deployers that need to build a single module from
        multiple nodes should call :meth:`emit` directly with the shared
        module context and then stringify the complete module themselves.

        Returns
        -------
        str
            MLIR text (printable module or fragment).
        """
        self.emit(operatorRepresentation, **kwargs)
        return ""
