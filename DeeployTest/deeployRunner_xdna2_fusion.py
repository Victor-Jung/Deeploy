#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper that runs the FusionRegion -> XDNA2 end-to-end test flow.

Usage (from DeeployTest/):
    python deeployRunner_xdna2_fusion.py -t Tests/Kernels/BF16/Relu/Regular
    python deeployRunner_xdna2_fusion.py -t Tests/Kernels/BF16/Relu/Regular --skipsim
"""

import sys

from testUtils.deeployRunner import main


def _add_xdna2_fusion_args(parser):
    parser.add_argument('--data-mode',
                        type = str,
                        choices = ['auto', 'embed', 'file'],
                        default = 'auto',
                        help = 'Where test inputs/outputs live: embed in header, .bin sidecars, '
                        'or auto-pick by size (default: auto).')


def _add_xdna2_fusion_gen_args(args, gen_args_list):
    data_mode = getattr(args, 'data_mode', 'auto')
    if data_mode != 'auto':
        gen_args_list.append(f'--data-mode={data_mode}')


if __name__ == '__main__':
    sys.exit(
        main(default_platform = "XDNA2",
             default_simulator = "host",
             tiling_enabled = False,
             platform_specific_cmake_args = [f"-DPython3_EXECUTABLE={sys.executable}"],
             parser_setup_callback = _add_xdna2_fusion_args,
             gen_args_callback = _add_xdna2_fusion_gen_args,
             generation_script = "generateFusionNetwork_xdna2.py"))
