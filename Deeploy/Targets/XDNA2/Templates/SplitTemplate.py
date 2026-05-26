# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for ``Split``.

Split is a layout transformation rather than a compute op, its MLIR
realisation is the ``aie.objectfifo.link`` distribute pattern produced by
:class:`MLIRDistributeLinkPass`. There is no compute-core kernel, so
``emit()`` is a no-op (no ``func.call`` to insert into a ``@aie_d.core``
body, because there is no ``@aie_d.core`` body for a Split node).

The N output keys (``data_out_0``..``data_out_{N-1}``) and the single
input key (``data_in``) match the keys :class:`SplitParser` populates in
the operator representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2SplitMemTileTemplate(MLIRNodeTemplate):
    """Mem-tile-hosted Split — pure data movement, no compute kernel."""

    KERNEL_FN = ""
    KERNEL_OBJ = ""
    INPUT_KEYS = ['data_in']
    # OUTPUT_KEYS is variadic per-instance; the deployer / link pass reads
    # data_out_{idx} from operatorRepresentation directly. Class default
    # left empty so passes that iterate it (e.g. trace passes) don't
    # mistakenly emit anything per output.
    OUTPUT_KEYS = []

    def __init__(self) -> None:
        super().__init__()

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        return  # no kernel call; link op is emitted by MLIRDistributeLinkPass


referenceTemplate = XDNA2SplitMemTileTemplate()
