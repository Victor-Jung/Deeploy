# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Fusion XDNA2 generation: <dir>/region.py -> MLIR-AIE (dedicated emitter).

`region.py` is the single committed source of truth for the test. Inputs
(random bf16) and the numpy golden are derived from it deterministically here,
so there are no committed data fixtures to go stale — edit region.py and re-run.
Emits `network.mlir` + `testinputs.h` / `testoutputs.h` for the XRT testbench.
"""

from __future__ import annotations

import os

import numpy as np

from Deeploy.FusionExtension.Reference import evaluate, random_inputs
from Deeploy.FusionExtension.Serialize import load_region
from Deeploy.Targets.XDNA2.Fusion.Emitter import emit
from testUtils.testRunner import TestGeneratorArgumentParser

from generateNetwork_xdna2 import _generate_xdna2_inputs_header, _generate_xdna2_outputs_header

_SEED = 0  # any fixed seed: inputs and golden are generated together, so correctness is seed-agnostic


def generateFusionNetworkXDNA2(args):
    region = load_region(f"{args.dir}/region.py")
    gin, gout = region.graph_inputs(), region.graph_outputs()

    # Derive data from the IR: random bf16 inputs + numpy golden (same order as
    # the emitter's runtime_sequence args = graph_inputs then graph_outputs).
    inputs = random_inputs(region, region.params, seed = _SEED)
    golden = evaluate(region, inputs)
    test_inputs = [inputs[n] for n in gin]
    test_outputs = [golden[n] for n in gout]

    os.makedirs(args.dumpdir, exist_ok = True)
    data_mode = getattr(args, "data_mode", "auto")
    input_padded = [int(np.prod(a.shape)) for a in test_inputs]
    output_padded = [int(np.prod(a.shape)) for a in test_outputs]

    header = _generate_xdna2_inputs_header(test_inputs, args.dumpdir, mode = data_mode, padded_elem_counts = input_padded)
    header += "#define TRACE_BUFFER_SIZE 0u\n"
    with open(f"{args.dumpdir}/testinputs.h", "w", encoding = "utf-8") as f:
        f.write(header)

    with open(f"{args.dumpdir}/testoutputs.h", "w", encoding = "utf-8") as f:
        f.write(
            _generate_xdna2_outputs_header(test_outputs, args.dumpdir, tolerance_ulps = 1, mode = data_mode,
                                           padded_elem_counts = output_padded))

    with open(f"{args.dumpdir}/network.mlir", "w", encoding = "utf-8") as f:
        f.write(emit(region))

    # Generic DeeployTest CMake expects a Network.c; XDNA2 ignores it.
    with open(f"{args.dumpdir}/Network.c", "w", encoding = "utf-8") as f:
        f.write("int deeploy_fusion_network_stub(void) { return 0; }\n")

    print(f"[FusionXDNA2] Emitted network.mlir + headers in {args.dumpdir}")


if __name__ == "__main__":
    parser = TestGeneratorArgumentParser(tiling_arguments = False,
                                         description = "Deeploy XDNA2 fusion code generation utility.")
    parser.add_argument('--data-mode', type = str, choices = ['auto', 'embed', 'file'], default = 'auto')
    args, _ = parser.parse_known_args()
    if args.platform != 'XDNA2':
        parser.error(f"This script is for the XDNA2 platform. Got: {args.platform}")
    generateFusionNetworkXDNA2(args)
