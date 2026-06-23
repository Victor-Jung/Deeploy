# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DSEConfig:
    """A single design point to evaluate."""

    # Operator / test identification
    test_dir: str  # path to test (e.g. Tests/Kernels/BF16/Add/Regular)
    op: str  # ONNX op name

    # Spatial mapping
    num_col: int = 1
    num_aie_row: int = 1

    # Memory budget (bytes)
    l1: int = 64000
    l2: int = 512 * 1024
    l3: Optional[int] = None

    # Extra CLI args forwarded to the runner
    extra_args: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_cores(self) -> int:
        return self.num_col * self.num_aie_row

    @property
    def label(self) -> str:
        return f"{self.op}_c{self.num_col}r{self.num_aie_row}"
