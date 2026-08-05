# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for IR-as-input fusion HW tests.

Each test folder holds a committed ``region.py`` (the source of truth). Tests
just run the folder on the NPU; inputs + golden are derived from region.py by
the generate step. ``write_region_py`` is authoring convenience for creating a
new test folder from a Python-built region.
"""

from __future__ import annotations

import os
import subprocess
import sys

from Deeploy.FusionExtension.IR import FusionRegion
from Deeploy.FusionExtension.Serialize import region_to_python

_DEEPLOYTEST = os.path.dirname(os.path.abspath(__file__))


def write_region_py(region: FusionRegion, abs_testdir: str) -> str:
    """Author a test folder: write the editable region.py source of truth."""
    os.makedirs(abs_testdir, exist_ok = True)
    path = os.path.join(abs_testdir, "region.py")
    with open(path, "w", encoding = "utf-8") as f:
        f.write(region_to_python(region))
    return path


def run_fusion_on_hw(rel_testdir: str) -> subprocess.CompletedProcess:
    """Invoke the fusion runner (region.py -> emit -> build -> NPU -> numeric check)."""
    env = dict(os.environ)
    env["LLVM_INSTALL_DIR"] = env.get("LLVM_INSTALL_DIR") or "nope"
    return subprocess.run([sys.executable, "deeployRunner_xdna2_fusion.py", "-t", rel_testdir],
                          cwd = _DEEPLOYTEST, env = env, capture_output = True, text = True)


def assert_hw_passed(result: subprocess.CompletedProcess) -> None:
    tail = f"\n--- stdout ---\n{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-1500:]}"
    assert result.returncode == 0, f"runner exited {result.returncode}{tail}"
    assert "PASSED" in result.stdout and "Errors: 0" in result.stdout, f"runner did not pass cleanly{tail}"
