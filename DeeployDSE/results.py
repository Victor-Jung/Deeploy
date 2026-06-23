# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LatencyStats:
    """Latency statistics from a hardware run (microseconds)."""
    min_us: float = 0.0
    max_us: float = 0.0
    mean_us: float = 0.0
    median_us: float = 0.0
    stdev_us: float = 0.0


@dataclass
class DSEResult:
    """Result of evaluating a single design point."""

    # Config reference
    op: str = ""
    num_col: int = 1
    num_aie_row: int = 1
    label: str = ""

    # Tensor info
    input_bytes: int = 0
    output_bytes: int = 0

    # Correctness
    errors: int = 0
    total_elems: int = 0
    passed: bool = False

    # Performance
    latency: LatencyStats = field(default_factory = LatencyStats)
    throughput_gbps: float = 0.0
    total_ops: int = 0

    # Build/run metadata
    build_success: bool = False
    error_message: Optional[str] = None

    @property
    def total_bytes(self) -> int:
        return self.input_bytes + self.output_bytes

    @property
    def total_cores(self) -> int:
        return self.num_col * self.num_aie_row
