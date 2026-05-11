# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Test list for the XDNA2 platform.

KERNEL_TESTS = [
    ("Kernels/BF16/Add/Regular", ["--num-col=1", "--num-aie-row=1"]),
    ("Kernels/BF16/Add/Regular", ["--num-col=2", "--num-aie-row=1"]),
    ("Kernels/BF16/Add/Regular", ["--num-col=4", "--num-aie-row=1"]),
    ("Kernels/BF16/Add/Regular", ["--num-col=8", "--num-aie-row=1"]),
    ("Kernels/BF16/SiLU/Regular", ["--num-col=1", "--num-aie-row=1"]),
    ("Kernels/BF16/SiLU/Regular", ["--num-col=2", "--num-aie-row=1"]),
    ("Kernels/BF16/SiLU/Regular", ["--num-col=4", "--num-aie-row=1"]),
    ("Kernels/BF16/SiLU/Regular", ["--num-col=8", "--num-aie-row=1"]),
]
