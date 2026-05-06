# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Test list for the XDNA2 platform.

KERNEL_TESTS = [
    "Kernels/BF16/Add/Regular",
    "Kernels/BF16/SiLU/Regular",
    "Kernels/BF16/LayerNorm/Regular",
]

SPATIAL_KERNEL_TESTS = [
    ("Kernels/BF16/Add/Regular", ["--num-cores=2"]),
]
