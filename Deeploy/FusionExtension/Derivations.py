# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Derivation functions over a :class:`FusionRegion` (pure, no lowering).

These are the numeric oracle: they let a scheme (scheme-D, regime-1, layer-by-
layer) be compared *before* any code is generated. See ``FUSION_IR_NOTES.md`` §7.

Corrections applied vs the handoff draft:
* ``offchip_elements`` includes a **region-boundary I/O** term — RESIDENT inputs
  are read once and RESIDENT accumulators are drained once (§7.3 as written only
  counted the STREAM refetch overhead and dropped the ``2TD`` in the FFN example).
* :func:`is_accumulator` requires the tensor to be **written** (a kernel output),
  distinguishing an accumulator from a read-only invariant input (both are
  RESIDENT and both may omit the REDUCE axis).
"""

from __future__ import annotations

from math import ceil
from typing import Dict, List

from Deeploy.FusionExtension.IR import EdgeSchedule, FusionRegion, LoopKind, Residency, TensorRef, dtype_bytes


# ---------------------------------------------------------------------------
# symbolic resolution
# ---------------------------------------------------------------------------


def resolve(sym, params: Dict[str, int]) -> int:
    """Resolve a symbolic extent/tile/dim to an int using ``params``.

    Accepts ints, integer-strings, plain names, or simple arithmetic like
    ``"64*R"`` / ``"4*D"`` (evaluated in a sandbox with ``params`` as names).
    """
    if isinstance(sym, int):
        return sym
    if isinstance(sym, str):
        try:
            return int(sym)
        except ValueError:
            pass
        return int(eval(sym, {"__builtins__": {}}, dict(params)))  # noqa: S307 — our own strings
    raise TypeError(f"cannot resolve symbolic value {sym!r}")


def _tensor_ref(region: FusionRegion, name: str) -> TensorRef:
    for k in region.kernels:
        for t in list(k.inputs) + list(k.outputs):
            if t.name == name:
                return t
    raise KeyError(f"tensor '{name}' not referenced by any kernel in the region")


def footprint_elements(region: FusionRegion, name: str) -> int:
    """Total element count of a tensor (product of its resolved full shape)."""
    ref = _tensor_ref(region, name)
    n = 1
    for d in ref.shape:
        n *= resolve(d, region.params)
    return n


def trip_count(region: FusionRegion, axis: str) -> int:
    loop = next((l for l in region.loops if l.axis == axis), None)
    if loop is None:
        return 1
    extent = resolve(loop.extent, region.params)
    tile = resolve(loop.tile, region.params)
    return max(1, ceil(extent / tile))


# ---------------------------------------------------------------------------
# structural queries
# ---------------------------------------------------------------------------


def is_accumulator(region: FusionRegion, edge: EdgeSchedule) -> bool:
    """RESIDENT + written by a kernel + its access omits every REDUCE axis."""
    return (edge.residency == Residency.RESIDENT and edge.tensor in region.output_names()
            and not (set(edge.access) & region.reduce_axes()))


def _compute_at_index(region: FusionRegion, name: str) -> int:
    """Index (in outer→inner loop order) of the loop level where the kernel
    touching ``name`` computes. Producer's placement if it's an output, else the
    first consumer's placement. Falls back to innermost."""
    order = [l.axis for l in region.loops]

    def _idx(axis: str) -> int:
        return order.index(axis) if axis in order else len(order)

    for k in region.kernels:
        if any(t.name == name for t in k.outputs):
            return _idx(region.placement.get(k.node, order[-1] if order else ""))
    for k in region.kernels:
        if any(t.name == name for t in k.inputs):
            return _idx(region.placement.get(k.node, order[-1] if order else ""))
    return len(order)


# ---------------------------------------------------------------------------
# the four derivation functions
# ---------------------------------------------------------------------------


def legality(region: FusionRegion) -> List[str]:
    """Return a list of legality violations (empty ⇒ legal).

    Implemented rule (§7.1): no REDUCE axis may appear in the ``access`` of any
    output tensor (it must be summed away). The dual rule ("no MARCH axis may be
    a reduction axis of any kernel") needs per-kernel reduction metadata not yet
    on the descriptor and is left as a TODO.
    """
    problems: List[str] = []
    red = region.reduce_axes()
    # Only the region's *terminal* outputs must have summed the reduction away.
    # Interior tensors (produced AND consumed here) may still carry a REDUCE axis
    # — they are mid-reduction (e.g. H in scheme-D is indexed by the fchunk it is
    # later contracted over).
    terminal_outputs = region.output_names() - region.input_names()
    for name in terminal_outputs:
        edge = region.edges.get(name)
        if edge is None:
            continue
        clash = set(edge.access) & red
        if clash:
            problems.append(f"terminal output '{name}' access contains REDUCE axis/axes {sorted(clash)} "
                            f"(a reduction axis must be summed away before it escapes the region)")
    return problems


def offchip_elements(region: FusionRegion) -> int:
    """Total DRAM traffic in *elements* (dtype-agnostic, matches §8 formulas).

    * TRANSIENT → 0 (never leaves the chip).
    * RESIDENT  → footprint once (region-boundary I/O: input read once, or
      accumulator drained once).
    * STREAM    → refetch × footprint, where refetch is the product of trip
      counts of loops OUTER to the tensor's compute-at level whose axis ∉ access.
    """
    total = 0
    produced = region.output_names()
    consumed = region.input_names()
    for name, edge in region.edges.items():
        fp = footprint_elements(region, name)
        if edge.residency == Residency.TRANSIENT:
            continue
        if edge.residency == Residency.RESIDENT:
            total += fp  # boundary I/O once (input read once, or accumulator drained once)
            continue
        # STREAM: a DRAM write if it is produced here, and a (refetched) DRAM read
        # if it is consumed here. An interior STREAM tensor pays both = round-trip.
        if name in produced:
            total += fp
        if name in consumed:
            ca = _compute_at_index(region, name)
            refetch = 1
            for i, loop in enumerate(region.loops):
                if i < ca and loop.axis not in edge.access:
                    refetch *= trip_count(region, loop.axis)
            total += refetch * fp
    return total


def offchip_bytes(region: FusionRegion) -> int:
    """Total DRAM traffic in bytes (per-tensor elements × dtype width)."""
    total = 0
    for name, edge in region.edges.items():
        elem = dtype_bytes(_tensor_ref(region, name).dtype)
        one = FusionRegion(kernels = region.kernels, loops = region.loops, placement = region.placement,
                           edges = {name: edge}, maps = region.maps, params = region.params, hw = region.hw)
        total += offchip_elements(one) * elem
    return total


def residency_footprint(region: FusionRegion) -> int:
    """Conservative on-chip footprint in bytes: the sum of the full-tensor
    footprints of every RESIDENT and TRANSIENT edge.

    NOTE: this is an *upper bound*. A precise peak-over-time (with per-tile
    residency for accumulators, e.g. the 64R×D panel rather than the full T×D
    output) is a refinement; :func:`accumulator_bytes` gives the exact per-panel
    accumulator size used by the on-chip-fits check.
    """
    total = 0
    for name, edge in region.edges.items():
        if edge.residency in (Residency.RESIDENT, Residency.TRANSIENT):
            total += footprint_elements(region, name) * dtype_bytes(_tensor_ref(region, name).dtype)
    return total


def accumulator_bytes(region: FusionRegion, name: str, acc_dtype_bytes: int = 4) -> int:
    """Exact on-chip size of a RESIDENT accumulator panel, in bytes.

    For each of the accumulator's dimensions: use the *tile* size if a loop axis
    maps to that dim (its resident slice is one tile), else the *full* extent.
    Accumulation is in fp32 by default (``acc_dtype_bytes=4``), matching the
    64R·D·4 corner in the notes.
    """
    ref = _tensor_ref(region, name)
    edge = region.edges[name]
    order = [l.axis for l in region.loops]
    red_idxs = [i for i, l in enumerate(region.loops) if l.kind == LoopKind.REDUCE]
    min_red = min(red_idxs) if red_idxs else len(order)
    # invert access: dim index -> loop axis
    dim_to_axis = {dim: axis for axis, dim in edge.access.items()}
    n = 1
    for dim_idx, d in enumerate(ref.shape):
        axis = dim_to_axis.get(dim_idx)
        # A dim is held at TILE size only if its axis is OUTER to every reduction
        # (the accumulator is reset per iteration of it). If the axis is INNER to
        # a reduction, the accumulator persists across all its values -> FULL.
        if axis is not None and axis in order and order.index(axis) < min_red:
            loop = next(l for l in region.loops if l.axis == axis)
            n *= resolve(loop.tile, region.params)
        else:
            n *= resolve(d, region.params)
    return n * acc_dtype_bytes
