# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Optional

from DeeployDSE.config import DSEConfig
from DeeployDSE.results import DSEResult, LatencyStats


class DSERunner:
    """Runs a single DSE config on hardware via deeployRunner_xdna2.py."""

    def __init__(self, deeploy_root: Optional[str] = None, python: Optional[str] = None):
        self.deeploy_root = deeploy_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.python = python or sys.executable
        self.test_root = os.path.join(self.deeploy_root, "DeeployTest")

    def run(self, config: DSEConfig) -> DSEResult:
        """Execute a design point and return parsed results."""
        cmd = self._build_cmd(config)
        result = DSEResult(
            op=config.op,
            num_col=config.num_col,
            num_aie_row=config.num_aie_row,
            label=config.label,
        )

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, cwd=self.test_root,
            )
            output = proc.stdout + proc.stderr
            result.build_success = proc.returncode == 0 or "Errors:" in output
            self._parse_output(output, result)
        except subprocess.TimeoutExpired:
            result.error_message = "Timeout (600s)"
        except Exception as e:
            result.error_message = str(e)

        return result

    def _build_cmd(self, config: DSEConfig) -> list:
        # test_dir may be relative to deeploy root; make it relative to test_root
        test_path = config.test_dir
        if test_path.startswith("DeeployTest/"):
            test_path = "./" + test_path[len("DeeployTest/"):]
        elif not test_path.startswith(("/", "./")):
            test_path = "./" + test_path

        cmd = [
            self.python, "deeployRunner_xdna2.py",
            "-t", test_path,
            "-v",
            f"--num-col={config.num_col}",
            f"--num-aie-row={config.num_aie_row}",
        ]
        if config.l1 != 64000:
            cmd.append(f"--l1={config.l1}")
        for k, v in config.extra_args.items():
            cmd.append(f"--{k}={v}")
        return cmd

    def _parse_output(self, output: str, result: DSEResult) -> None:
        # Parse errors line: "Errors: 0 out of 102400"
        m = re.search(r"Errors:\s+(\d+)\s+out of\s+(\d+)", output)
        if m:
            result.errors = int(m.group(1))
            result.total_elems = int(m.group(2))
            result.passed = result.errors == 0

        # Parse latency: "latency [us]  : min=X  median=Y  mean=Z  max=W  stdev=S"
        m = re.search(
            r"latency \[us\]\s*:\s*min=([\d.]+)\s+median=([\d.]+)\s+mean=([\d.]+)"
            r"\s+max=([\d.]+)\s+stdev=([\d.]+)", output)
        if m:
            result.latency = LatencyStats(
                min_us=float(m.group(1)),
                median_us=float(m.group(2)),
                mean_us=float(m.group(3)),
                max_us=float(m.group(4)),
                stdev_us=float(m.group(5)),
            )

        # Parse throughput: "throughput estimate  : X.XXX GB/s (NNNN bytes / median latency)"
        m = re.search(r"throughput.*?:\s+([\d.]+)\s+GB/s\s+\((\d+)\s+bytes", output)
        if m:
            result.throughput_gbps = float(m.group(1))
            total_bytes = int(m.group(2))
            result.input_bytes = total_bytes // 2  # rough split
            result.output_bytes = total_bytes - result.input_bytes

        # Parse total ops: "[XDNA2] Total Operations: N" or "Total Operations: N"
        m = re.search(r"Total Operations:\s+(\d+)", output)
        if m:
            result.total_ops = int(m.group(1))
