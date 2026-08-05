# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Fusion IR — target-agnostic data structures.

Two layers (see ``FUSION_IR_NOTES.md`` §7):

* **Layer 1 — descriptors** (:class:`BoundKernel`, :class:`TensorRef`): what each
  kernel *is*, produced by Deeploy's elect-only path (parse + typeCheck, no bind).
  Tiling-agnostic: kernel symbol + parsed attrs + typed I/O + a *reference* to a
  TileConstraint, but no tile sizes and no placement.
* **Layer 2 — structure** (:class:`FusionRegion` and friends): how a group of
  kernels is woven into one schedule — the shared loop nest, where each interior
  tensor lives (:class:`EdgeSchedule`), and how output tiles map onto the PE grid
  (:class:`SpatialMap`).

The hardware is *injected* via :class:`HardwareModel`, which is mostly derivable
from the Deeploy platform's execution engines + memory hierarchy — so the IR
stays agnostic to any specific accelerator geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Type

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
}


def dtype_bytes(dtype: str) -> int:
    """Byte width of an IR dtype string."""
    key = dtype.lower()
    if key not in _DTYPE_BYTES:
        raise KeyError(f"unknown IR dtype '{dtype}'; known: {sorted(_DTYPE_BYTES)}")
    return _DTYPE_BYTES[key]


# ---------------------------------------------------------------------------
# Layer 1 — descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen = True)
class TensorRef:
    """A typed tensor reference. ``shape`` entries may be ints or symbolic
    names (resolved against :attr:`FusionRegion.params`)."""
    name: str
    shape: Tuple[Any, ...]
    dtype: str


@dataclass
class BoundKernel:
    """A single elected kernel — the frontend↔IR contract (freeze this).

    Carries everything the IR needs *except* tiling and placement: the AIE
    kernel symbol, the parser's operator representation, typed I/O, and a
    reference (not an instance) to the Deeploy TileConstraint the tiler will use.
    """
    node: str
    op: str
    kernel_symbol: str
    operator_repr: Dict[str, Any]
    inputs: List[TensorRef]
    outputs: List[TensorRef]
    tile_constraint: Optional[Type] = None
    kernel_obj: Optional[str] = None  # link `.o` for the AIE kernel (needed for lowering)


# ---------------------------------------------------------------------------
# Layer 2 — structure
# ---------------------------------------------------------------------------


class LoopKind(Enum):
    MARCH = auto()     #: output axis walked in time (waves)
    REDUCE = auto()    #: contraction axis, summed away (absent from outputs)
    PARALLEL = auto()  #: output axis laid out spatially across the PE grid


class Residency(Enum):
    STREAM = auto()     #: flows to/from DRAM
    TRANSIENT = auto()  #: on-chip scratch, never touches DRAM (the fusion knob)
    RESIDENT = auto()   #: lives on-chip across the loop (invariant input or accumulator)


@dataclass
class Loop:
    """One axis of the shared loop nest. ``extent``/``tile`` are symbolic
    (resolved via :attr:`FusionRegion.params`)."""
    axis: str
    extent: str
    tile: str
    kind: LoopKind


@dataclass
class EdgeSchedule:
    """Where a tensor lives and how the loop nest indexes it.

    ``access`` maps a loop axis name → the tensor dimension index it walks
    (e.g. ``{"m": 0, "k": 1}`` for A[M,K]). The *set of keys* drives the
    residency/accumulator/reuse logic; the *values* give the concrete tile slice.
    """
    tensor: str
    access: Dict[str, int]
    residency: Residency
    level: str


@dataclass
class SpatialMap:
    """How one kernel's output tiles map onto the physical PE grid.

    ``row_axis``/``col_axis`` point at loop axes assigned to PE rows/columns;
    ``stationary`` is the output-stationary tensor; ``core_range`` is the (R, C)
    PE sub-grid (ints or symbolic names)."""
    node: str
    row_axis: Optional[str]
    col_axis: Optional[str]
    stationary: Optional[str]
    core_range: Tuple[Any, Any]


@dataclass
class HardwareModel:
    """Injected hardware description — mostly derivable from the Deeploy
    platform's execution engines and memory hierarchy.

    Keeping this explicit (rather than hard-coding XDNA2 numbers) is what makes
    the derivation functions and spatial-map lowering target-agnostic.
    """
    pe_rows: int
    pe_cols: int
    pe_block: Tuple[int, int] = (64, 64)          #: output block per PE (rows, cols)
    mem_capacity: Dict[str, int] = field(default_factory = dict)  #: level name → bytes

    @classmethod
    def from_platform(cls, platform: Any) -> "HardwareModel":
        """Build a :class:`HardwareModel` from a Deeploy deployment platform by
        reading its execution engines (PE grid) and memory hierarchy (capacities).

        Only relies on duck-typed attributes (``.engines`` with ``.col``/``.row``,
        ``.memoryHierarchy.memoryLevels`` with ``.size``) so it works for any
        platform that exposes an AIE-like engine set.
        """
        cols: set = set()
        rows: set = set()
        for eng in getattr(platform, "engines", []):
            col = getattr(eng, "col", None)
            row = getattr(eng, "row", None)
            if col is not None:
                cols.add(int(col))
            if row is not None:
                rows.add(int(row))

        mem_capacity: Dict[str, int] = {}
        hierarchy = getattr(platform, "memoryHierarchy", None)
        levels = getattr(hierarchy, "memoryLevels", {}) if hierarchy is not None else {}
        for name, level in levels.items():
            size = getattr(level, "size", None)
            if size is not None:
                mem_capacity[name] = int(size)

        return cls(pe_rows = max(len(rows), 1), pe_cols = max(len(cols), 1), mem_capacity = mem_capacity)


@dataclass
class FusionRegion:
    """A fusion group: descriptors + the schedule that welds them together."""
    kernels: List[BoundKernel]
    loops: List[Loop]                    #: outer-to-inner
    placement: Dict[str, str]            #: node name → loop axis (compute-at)
    edges: Dict[str, EdgeSchedule]       #: tensor name → schedule
    maps: Dict[str, SpatialMap] = field(default_factory = dict)  #: node name → spatial map
    params: Dict[str, int] = field(default_factory = dict)       #: symbolic extent → value
    hw: Optional[HardwareModel] = None

    # -- convenience accessors --------------------------------------------
    def kernel_by_node(self, node: str) -> Optional[BoundKernel]:
        return next((k for k in self.kernels if k.node == node), None)

    def reduce_axes(self) -> set:
        return {loop.axis for loop in self.loops if loop.kind == LoopKind.REDUCE}

    def output_names(self) -> set:
        return {t.name for k in self.kernels for t in k.outputs}

    def input_names(self) -> set:
        return {t.name for k in self.kernels for t in k.inputs}

    def interior_names(self) -> set:
        """Tensors that are both produced and consumed inside the region."""
        return self.output_names() & self.input_names()

    def graph_inputs(self) -> List[str]:
        """Region inputs (consumed, not produced), in first-appearance order."""
        produced, seen, out = self.output_names(), set(), []
        for k in self.kernels:
            for t in k.inputs:
                if t.name not in produced and t.name not in seen:
                    seen.add(t.name)
                    out.append(t.name)
        return out

    def graph_outputs(self) -> List[str]:
        """Region outputs (produced, not consumed), in first-appearance order."""
        consumed, seen, out = self.input_names(), set(), []
        for k in self.kernels:
            for t in k.outputs:
                if t.name not in consumed and t.name not in seen:
                    seen.add(t.name)
                    out.append(t.name)
        return out
