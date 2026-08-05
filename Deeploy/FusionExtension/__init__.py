# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Fusion IR extension: target-agnostic IR for deep operator fusion.

Only the pure, backend-free layer is re-exported here (IR + derivations +
reference). XDNA2 lowering lives in ``Deeploy.Targets.XDNA2.Fusion`` and must be
imported explicitly, so importing this package never drags in the deployer stack.

See ``FUSION_IR_NOTES.md`` (design) and ``FUSION_IR_PLAN.md`` (plan).
"""

from Deeploy.FusionExtension.Derivations import accumulator_bytes, is_accumulator, legality, offchip_bytes, \
    offchip_elements, residency_footprint, resolve
from Deeploy.FusionExtension.IR import BoundKernel, EdgeSchedule, FusionRegion, HardwareModel, Loop, LoopKind, \
    Residency, SpatialMap, TensorRef, dtype_bytes

__all__ = [
    "BoundKernel", "EdgeSchedule", "FusionRegion", "HardwareModel", "Loop", "LoopKind", "Residency", "SpatialMap",
    "TensorRef", "dtype_bytes", "accumulator_bytes", "is_accumulator", "legality", "offchip_bytes", "offchip_elements",
    "residency_footprint", "resolve"
]
