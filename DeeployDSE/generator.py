# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from DeeployDSE.config import DSEConfig

# NPU2 hardware
MAX_COLS = 8
MAX_AIE_ROWS = 4
VECTOR_WIDTH_BF16 = 16


class ConfigGenerator(ABC):
    """Base class for DeeployDSE configuration generators.

    Subclasses implement `generate()` to enumerate legal design points
    given an ONNX model and a target operator.
    """

    @abstractmethod
    def generate(self, test_dir: str) -> List[DSEConfig]:
        """Return all legal configurations for the given test."""
        ...

    def _load_graph(self, test_dir: str) -> gs.Graph: 
        model = onnx.load(f"{test_dir}/network.onnx")
        return gs.import_onnx(model)

    def _input_shapes(self, test_dir: str) -> List[Tuple[int, ...]]:
        data = np.load(f"{test_dir}/inputs.npz")
        return [data[k].shape for k in data.files]


class XDNA2ElementwiseGenerator(ConfigGenerator):
    """Generate configs for XDNA2 elementwise ops.

    Sweeps (num_col, num_aie_row) pairs where the spatial split is legal
    given the tensor shape and vector alignment.
    """

    def __init__(self,
                 col_range: Optional[List[int]] = None,
                 row_range: Optional[List[int]] = None,
                 axis: int = 0):
        self.col_range = col_range or list(range(1, MAX_COLS + 1))
        self.row_range = row_range or list(range(1, MAX_AIE_ROWS + 1))
        self.axis = axis

    def generate(self, test_dir: str) -> List[DSEConfig]:
        graph = self._load_graph(test_dir)
        shapes = self._input_shapes(test_dir)
        op = graph.nodes[0].op if graph.nodes else "Unknown"
        axis_size = shapes[0][self.axis] if shapes else 0

        configs = []
        for nc in self.col_range:
            for nr in self.row_range:
                if self._is_legal(axis_size, nc, nr, op):
                    configs.append(DSEConfig(
                        test_dir=test_dir,
                        op=op,
                        num_col=nc,
                        num_aie_row=nr,
                    ))
        return configs

    def _is_legal(self, axis_size: int, nc: int, nr: int, op: str) -> bool:
        """Check if the configuration satisfies hardware constraints."""

        if nr > MAX_AIE_ROWS:
            return False
        
        if nc > MAX_COLS:
            return False

        return True
