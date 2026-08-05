# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Numpy golden reference for a FusionRegion (target-agnostic).

Evaluates a region op-by-op in bf16 so every E2E slice has an expected output
without hand-written data. bf16 truncation matches the XDNA2 kernels' storage.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from ml_dtypes import bfloat16 as _bf16

from Deeploy.FusionExtension.IR import FusionRegion

_OPS = {
    "Relu": lambda ins: np.maximum(ins[0], 0.0),
    "Add": lambda ins: ins[0] + ins[1],
    "Mul": lambda ins: ins[0] * ins[1],
}


def _t(x: np.ndarray) -> np.ndarray:
    return x.astype(_bf16).astype(np.float32)


def evaluate(region: FusionRegion, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map {graph_input_name: array} -> {graph_output_name: array}, all bf16-truncated."""
    env = {k: _t(v) for k, v in inputs.items()}
    for k in region.kernels:
        if k.op not in _OPS:
            raise NotImplementedError(f"no numpy reference for op '{k.op}'")
        env[k.outputs[0].name] = _t(_OPS[k.op]([env[t.name] for t in k.inputs]))
    return {name: env[name] for name in region.graph_outputs()}


def random_inputs(region: FusionRegion, params: Dict[str, int], seed: int = 0) -> Dict[str, np.ndarray]:
    """Random bf16 inputs (as float32) for each graph input, shaped from the IR."""
    from Deeploy.FusionExtension.Derivations import resolve
    rng = np.random.default_rng(seed)
    refs = {t.name: t for kk in region.kernels for t in kk.inputs}
    out: Dict[str, np.ndarray] = {}
    for name in region.graph_inputs():
        shape = tuple(resolve(d, params) for d in refs[name].shape)
        out[name] = _t(rng.standard_normal(shape).astype(np.float32))
    return out
