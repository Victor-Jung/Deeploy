# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""E2E fusion test: temporal fusion of Y = relu(A + B) on a single core.

The IR is the committed ``Tests/Fusion/TemporalAddRelu/region.py``. Its interior
`s` is TRANSIENT/L1, so the emitter chains both kernels in one loop over a
core-local scratch buffer and `s` never round-trips to DRAM. Built and run on
the XDNA2 NPU against a numpy golden derived from region.py.
"""

import os

import pytest

from fusionTestUtils import assert_hw_passed, run_fusion_on_hw


@pytest.mark.xdna2
def test_temporal_add_relu_fusion_on_hw():
    assert_hw_passed(run_fusion_on_hw(os.path.join("Tests", "Fusion", "TemporalAddRelu")))
