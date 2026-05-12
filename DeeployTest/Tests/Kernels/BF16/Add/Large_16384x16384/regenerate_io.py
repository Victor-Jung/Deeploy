#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""Regenerate inputs.npz and outputs.npz from network.onnx in this directory.

The .npz files for this test are gitignored (they're ~1 GB each); this
script reproduces them deterministically from the checked-in network.onnx
using the shared reference implementation in ``_generate_elementwise.py``.

Run once after checkout (or whenever network.onnx changes)::

    python regenerate_io.py
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
# .../Tests/Kernels/BF16 is two levels up from <Op>/<Variant>/
sys.path.insert(0, str(HERE.parent.parent))
from _generate_elementwise import regenerate_io_from_onnx  # noqa: E402

regenerate_io_from_onnx(HERE / "network.onnx", HERE)
