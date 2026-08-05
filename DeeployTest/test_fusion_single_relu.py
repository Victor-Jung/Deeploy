# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""E2E fusion test: single-node Relu.

The IR is the committed ``Tests/Fusion/SingleRelu/region.py`` (source of truth).
It is lowered by the dedicated IR->MLIR emitter (no synthetic graph / tiler),
built, and run on the XDNA2 NPU against a numpy golden derived from region.py.
Exercises the emitter's single-core, single-kernel path.
"""

import os

import pytest

from fusionTestUtils import assert_hw_passed, run_fusion_on_hw


@pytest.mark.xdna2
def test_single_relu_on_hw():
    assert_hw_passed(run_fusion_on_hw(os.path.join("Tests", "Fusion", "SingleRelu")))
