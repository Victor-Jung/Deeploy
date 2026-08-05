# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Serialize a FusionRegion to/from a human-readable, editable Python file.

The emitted file constructs a ``region`` object with plain constructor calls, so
it can be read and hand-edited in a test folder and re-run through the runner.
``load_region`` execs such a file and returns its ``region``.
"""

from __future__ import annotations

from dataclasses import MISSING, is_dataclass
from enum import Enum
from typing import Any

from Deeploy.FusionExtension.IR import FusionRegion

_HEADER = '''# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Fusion IR for this test - human-readable and directly editable.

Edit `region` and re-run the fusion runner on this folder to re-emit the MLIR.
"""

from Deeploy.FusionExtension.IR import BoundKernel, EdgeSchedule, FusionRegion, Loop, LoopKind, Residency, TensorRef
'''


def _fmt(v: Any, indent: int = 0) -> str:
    pad = "    " * indent
    if isinstance(v, Enum):
        return f"{type(v).__name__}.{v.name}"
    if is_dataclass(v) and not isinstance(v, type):
        cls = type(v).__name__
        # skip fields left at their (non-factory) default to cut noise
        fields = [
            f for f, spec in v.__dataclass_fields__.items()
            if not (spec.default is not MISSING and getattr(v, f) == spec.default)
        ]
        # multi-line for the big records, one-line for the small ones
        if cls in ("BoundKernel", "FusionRegion"):
            inner = ",\n".join(f"{pad}    {f} = {_fmt(getattr(v, f), indent + 1)}" for f in fields)
            return f"{cls}(\n{inner},\n{pad})"
        args = ", ".join(_fmt(getattr(v, f), indent) for f in fields)
        return f"{cls}({args})"
    if isinstance(v, dict):
        if not v:
            return "{}"
        if len(v) > 2:  # break large dicts (e.g. edges) one entry per line
            inner = ",\n".join(f"{pad}    {_fmt(k)}: {_fmt(val, indent + 1)}" for k, val in v.items())
            return "{\n" + inner + f",\n{pad}}}"
        items = ", ".join(f"{_fmt(k)}: {_fmt(val, indent)}" for k, val in v.items())
        return "{" + items + "}"
    if isinstance(v, tuple):
        body = ", ".join(_fmt(x, indent) for x in v)
        return f"({body},)" if len(v) == 1 else f"({body})"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x, indent) for x in v) + "]"
    return repr(v)


def region_to_python(region: FusionRegion) -> str:
    """Render ``region`` as importable Python source defining ``region``."""
    if any(k.tile_constraint is not None for k in region.kernels):
        # tile_constraint is a type reference; not round-tripped yet (always None in current tests).
        raise NotImplementedError("region_to_python does not serialize tile_constraint yet")
    return f"{_HEADER}\nregion = {_fmt(region)}\n"


def load_region(path: str) -> FusionRegion:
    """Exec a region.py file and return its ``region`` (trusted local file)."""
    ns: dict = {}
    with open(path, encoding = "utf-8") as f:
        exec(compile(f.read(), path, "exec"), ns)  # noqa: S102 - trusted test artifact
    region = ns.get("region")
    if not isinstance(region, FusionRegion):
        raise ValueError(f"{path} did not define a `region` FusionRegion")
    return region
